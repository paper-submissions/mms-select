import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.nn.utils import clip_grad_norm_
from utils.meters import AverageMeter, accuracy
from utils.mixup import MixUp, CutMix
from random import sample
try:
    import tensorwatch
    _TENSORWATCH_AVAILABLE = True
except ImportError:
    _TENSORWATCH_AVAILABLE = False


def _flatten_duplicates(inputs, target, batch_first=True, expand_target=True):
    duplicates = inputs.size(1)
    if not batch_first:
        inputs = inputs.transpose(0, 1)
    inputs = inputs.flatten(0, 1)

    if expand_target:
        if batch_first:
            target = target.view(-1, 1).expand(-1, duplicates)
        else:
            target = target.view(1, -1).expand(duplicates, -1)
        target = target.flatten(0, 1)
    return inputs, target


def _average_duplicates(outputs, target, batch_first=True):
    """assumes target is not expanded (target.size(0) == batch_size) """
    batch_size = target.size(0)
    reduce_dim = 1 if batch_first else 0
    if batch_first:
        outputs = outputs.view(batch_size, -1, *outputs.shape[1:])
    else:
        outputs = outputs.view(-1, batch_size, *outputs.shape[1:])
    outputs = outputs.mean(dim=reduce_dim)
    return outputs


def _mixup(mixup_modules, alpha, batch_size):
    mixup_layer = None
    if len(mixup_modules) > 0:
        for m in mixup_modules:
            m.reset()
        mixup_layer = sample(mixup_modules, 1)[0]
        mixup_layer.sample(alpha, batch_size)
    return mixup_layer


class Trainer(object):

    def __init__(self, model, criterion, optimizer=None, calc_grad_var=None,
                 device_ids=[0], device=torch.cuda, dtype=torch.float,
                 distributed=False, local_rank=-1, adapt_grad_norm=None, 
                 mixup=None, cutmix=None, loss_scale=1., grad_clip=-1, print_freq=100,
                 batch_size=64):
        self._model = model
        self.criterion = criterion
        self.epoch = 0
        self.training_steps = 0
        self.optimizer = optimizer
        self.device = device
        self.dtype = dtype
        self.distributed = distributed
        self.local_rank = local_rank
        self.print_freq = print_freq
        self.grad_clip = grad_clip
        self.mixup = mixup
        self.cutmix = cutmix
        self.batch_size = batch_size
        self.grad_scale = None
        self.loss_scale = loss_scale
        self.adapt_grad_norm = adapt_grad_norm
        self.calc_grad_var = calc_grad_var
        self.watcher = None
        self.streams = {}

        if distributed:
            self.model = nn.parallel.DistributedDataParallel(model,
                                                             device_ids=device_ids,
                                                             output_device=device_ids[0])
        elif device_ids and len(device_ids) > 1:
            self.model = nn.DataParallel(model, device_ids)
        else:
            self.model = model

    def _grad_norm(self, inputs_batch, target_batch, chunk_batch=1):
        self.model.zero_grad()
        for inputs, target in zip(inputs_batch.chunk(chunk_batch, dim=0),
                                  target_batch.chunk(chunk_batch, dim=0)):
            target = target.to(self.device)
            inputs = inputs.to(self.device, dtype=self.dtype)

            # compute output
            output = self.model(inputs)
            loss = self.criterion(output, target)

            if chunk_batch > 1:
                loss = loss / chunk_batch

            loss.backward()   # accumulate gradient
        grad = clip_grad_norm_(self.model.parameters(), float('inf'))
        return grad

    def _step(self, inputs_batch, target_batch, training=False, average_output=False, chunk_batch=1):
        outputs = []
        total_loss = 0

        if training:
            self.optimizer.zero_grad()
            self.optimizer.update(self.epoch, self.training_steps)

        for i, (inputs, target) in enumerate(zip(inputs_batch.chunk(chunk_batch, dim=0),
                                                 target_batch.chunk(chunk_batch, dim=0))):
            target = target.to(self.device)
            inputs = inputs.to(self.device, dtype=self.dtype)

            mixup = None
            if training:
                self.optimizer.pre_forward()
                if self.mixup is not None or self.cutmix is not None:
                    input_mixup = CutMix() if self.cutmix else MixUp()
                    mix_val = self.mixup or self.cutmix
                    mixup_modules = [input_mixup]  # input mixup
                    mixup_modules += [m for m in self.model.modules()
                                      if isinstance(m, MixUp)]
                    mixup = _mixup(mixup_modules, mix_val, inputs.size(0))
                    inputs = input_mixup(inputs)

            # compute output
            output, _ = self.model(inputs)

            if mixup is not None:
                target = mixup.mix_target(target, output.size(-1))

            if average_output:
                if isinstance(output, list) or isinstance(output, tuple):
                    output = [_average_duplicates(out, target) if out is not None else None
                              for out in output]
                else:
                    output = _average_duplicates(output, target)
            loss = self.criterion(output, target)
            grad = None

            if chunk_batch > 1:
                loss = loss / chunk_batch

            if isinstance(output, list) or isinstance(output, tuple):
                output = output[0]

            outputs.append(output.detach())
            total_loss += float(loss)

            if training:
                if i == 0:
                    self.optimizer.pre_backward()
                if self.grad_scale is not None:
                    loss = loss * self.grad_scale
                if self.loss_scale is not None:
                    loss = loss * self.loss_scale
                loss.backward()   # accumulate gradient

        if training:  # post gradient accumulation
            if self.loss_scale is not None:
                for p in self.model.parameters():
                    if p.grad is None:
                        continue
                    p.grad.data.div_(self.loss_scale)

            if self.grad_clip > 0:
                grad = clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()  # SGD step
            self.training_steps += 1

        outputs = torch.cat(outputs, dim=0)
        return outputs, total_loss, grad

    def forward(self, data_loader, num_steps=None, training=False, average_output=False, chunk_batch=1):

        meters = {name: AverageMeter()
                  for name in ['step', 'data', 'loss', 'prec1', 'prec5']}
        if training and self.grad_clip > 0:
            meters['grad'] = AverageMeter()

        batch_first = True
        if training and isinstance(self.model, nn.DataParallel) or chunk_batch > 1:
            batch_first = False

        def meter_results(meters):
            results = {name: meter.avg for name, meter in meters.items()}
            results['error1'] = 100. - results['prec1']
            results['error5'] = 100. - results['prec5']
            return results

        end = time.time()

        for i, (inputs, target) in enumerate(data_loader):
            duplicates = inputs.dim() > 4  # B x D x C x H x W
            if training and duplicates and self.adapt_grad_norm is not None \
                    and i % self.adapt_grad_norm == 0:
                grad_mean = 0
                num = inputs.size(1)
                for j in range(num):
                    grad_mean += float(self._grad_norm(inputs.select(1, j), target))
                grad_mean /= num
                grad_all = float(self._grad_norm(
                    *_flatten_duplicates(inputs, target, batch_first)))
                self.grad_scale = grad_mean / grad_all
                logging.info('New loss scale: %s', self.grad_scale)

            # measure data loading time
            meters['data'].update(time.time() - end)
            if duplicates:  # multiple versions for each sample (dim 1)
                inputs, target = _flatten_duplicates(inputs, target, batch_first,
                                                     expand_target=not average_output)

            output, loss, grad = self._step(inputs, target,
                                            training=training,
                                            average_output=average_output,
                                            chunk_batch=chunk_batch)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            meters['loss'].update(float(loss), inputs.size(0))
            meters['prec1'].update(float(prec1), inputs.size(0))
            meters['prec5'].update(float(prec5), inputs.size(0))
            if grad is not None:
                meters['grad'].update(float(grad), inputs.size(0))

            # measure elapsed time
            meters['step'].update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0 or i == len(data_loader) - 1:
                report = str('{phase} - Epoch: [{0}][{1}/{2}]\t'
                             'Time {meters[step].val:.3f} ({meters[step].avg:.3f})\t'
                             'Data {meters[data].val:.3f} ({meters[data].avg:.3f})\t'
                             'Loss {meters[loss].val:.4f} ({meters[loss].avg:.4f})\t'
                             'Prec@1 {meters[prec1].val:.3f} ({meters[prec1].avg:.3f})\t'
                             'Prec@5 {meters[prec5].val:.3f} ({meters[prec5].avg:.3f})\t'
                             .format(
                                 self.epoch, i, len(data_loader),
                                 phase='TRAINING' if training else 'EVALUATING',
                                 meters=meters))
                if 'grad' in meters.keys():
                    report += 'Grad {meters[grad].val:.3f} ({meters[grad].avg:.3f})'\
                        .format(meters=meters)
                logging.info(report)
                self.observe(trainer=self,
                             model=self._model,
                             optimizer=self.optimizer,
                             data=(inputs, target))
                self.stream_meters(meters,
                                   prefix='train' if training else 'eval')
                if training:
                    self.write_stream('lr',
                                      (self.training_steps, self.optimizer.get_lr()[0]))

            if num_steps is not None and i >= num_steps:
                break

        return meter_results(meters)

    def train(self, data_loader, average_output=False, chunk_batch=1):
        # switch to train mode
        self.model.train()
        self.write_stream('epoch', (self.training_steps, self.epoch))
        return self.forward(data_loader, training=True, average_output=average_output, chunk_batch=chunk_batch)

    def validate(self, data_loader, average_output=False):
        # switch to evaluate mode
        self.model.eval()
        with torch.no_grad():
            return self.forward(data_loader, average_output=average_output, training=False)

    def calibrate_bn(self, data_loader, num_steps=None):
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.momentum = None
                m.track_running_stats = True
                m.reset_running_stats()
        self.model.train()
        with torch.no_grad():
            return self.forward(data_loader, num_steps=num_steps, training=False)

    ###### tensorwatch methods to enable training-time logging ######

    def set_watcher(self, filename, port=0):
        if not _TENSORWATCH_AVAILABLE:
            return False
        if self.distributed and self.local_rank > 0:
            return False
        self.watcher = tensorwatch.Watcher(filename=filename, port=port)
        # default streams
        self._default_streams()
        self.watcher.make_notebook()
        return True

    def get_stream(self, name, **kwargs):
        if self.watcher is None:
            return None
        if name not in self.streams.keys():
            self.streams[name] = self.watcher.create_stream(name=name,
                                                            **kwargs)
        return self.streams[name]

    def write_stream(self, name, values):
        stream = self.get_stream(name)
        if stream is not None:
            stream.write(values)

    def stream_meters(self, meters_dict, prefix=None):
        if self.watcher is None:
            return False
        for name, value in meters_dict.items():
            if prefix is not None:
                name = '_'.join([prefix, name])
            value = value.val
            stream = self.get_stream(name)
            if stream is None:
                continue
            stream.write((self.training_steps, value))
        return True

    def observe(self, **kwargs):
        if self.watcher is None:
            return False
        self.watcher.observe(**kwargs)
        return True

    def _default_streams(self):
        self.get_stream('train_loss')
        self.get_stream('eval_loss')
        self.get_stream('train_prec1')
        self.get_stream('eval_prec1')
        self.get_stream('lr')
        
       
class SelectionTrainer(Trainer):

    def __init__(self, *kargs,  **kwargs):
        super(SelectionTrainer, self).__init__(*kargs,  **kwargs)
        self.data_loader = kwargs.pop('data_loader', None)
        self.ratio = kwargs.pop('ratio', 10)
        

    def select_largest_entropy(self, inputs, target):
        self.model.eval()
        with torch.no_grad():
            inputs = inputs.to(self.device, dtype=self.dtype)
            logit, _ = self.model(inputs)
            # logit = logit.sigmoid()
            # logit = logit / logit.sum(-1).unsqueeze(-1)
            # log_logit = torch.log(logit)
            log_prob_bins = nn.functional.log_softmax(logit, 1)
            prob_bins = torch.softmax(logit, 1)

            entropy_logit = torch.sum(-prob_bins * log_prob_bins, dim=1)

            _, max_entr_indices = torch.sort(entropy_logit, descending=True)

            max_entr_indices = max_entr_indices[:self.batch_size]
        self.model.train()
        return inputs[max_entr_indices], target[max_entr_indices]
    
    def select_mms(self, inputs, target, meters):
        self.model.eval()
        with torch.no_grad():
            inputs = inputs.to(self.device, dtype=self.dtype)
            target = target.to(self.device)
            logit, _ = self.model(inputs)
            topk_vals, pred_classes = logit.topk(2, 1)


            w1 = self.model.fc.weight.index_select(0, pred_classes[:, 0])
            w2 = self.model.fc.weight.index_select(0, pred_classes[:, 1])
            entropy_logit = (topk_vals[:,0] - topk_vals[:,1]) / (w1 - w2).norm(dim=-1)

            _, min_mms_indices = torch.sort(entropy_logit)

            confidence_top10 = torch.mean(entropy_logit[min_mms_indices][:10])
            meters['confidence'].update(float(confidence_top10), 1)
            
            min_mms_indices = min_mms_indices[:self.batch_size]
        self.model.train()
        return inputs[min_mms_indices], target[min_mms_indices] 




    def forward(self, data_loader, num_steps=None, training=False, average_output=False, chunk_batch=1):
        self.train_batches = len(data_loader)
        meters = {name: AverageMeter()
                  for name in ['step', 'data', 'loss', 'prec1', 'prec5', 'samples', 'confidence']}
        if training and self.grad_clip > 0:
            meters['grad'] = AverageMeter()
        if self.calc_grad_var is not None:
            var_meter = OnlineMeter()
            meters['grad_var'] = AverageMeter()

        batch_first = True
        if training and isinstance(self.model, nn.DataParallel) or chunk_batch > 1:
            batch_first = False

        def meter_results(meters):
            results = {name: meter.avg for name, meter in meters.items()}
            results['error1'] = 100. - results['prec1']
            results['error5'] = 100. - results['prec5']
            return results

        end = time.time()

        start_lr = self.optimizer.get_lr()[0]
        for i, (inputs, target) in enumerate(data_loader):

            # measure data loading time
            meters['data'].update(time.time() - end)

            
            inputs, target = self.select_mms(inputs, target, meters)
            

            output, loss, grad = self._step(inputs, target,
                                            training=training,
                                            average_output=average_output,
                                            chunk_batch=chunk_batch)

            if self.calc_grad_var is not None:
                var_meter.update(self.collect_flatten_grads_(self.model.parameters()))
                if (self.training_steps + 1) % self.calc_grad_var == 0:
                    meters['grad_var'].update(float(var_meter.var.mean()), inputs.size(0))
                    var_meter.needs_init = True

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            meters['loss'].update(float(loss), inputs.size(0))
            meters['prec1'].update(float(prec1), inputs.size(0))
            meters['prec5'].update(float(prec5), inputs.size(0))
            # self.data_idx_set.update(set(bindices.tolist()))
            # meters['samples'].update(int(len(self.data_idx_set)), 1)
            if grad is not None:
                meters['grad'].update(float(grad), inputs.size(0))

            # measure elapsed time
            meters['step'].update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0 or i == len(data_loader) - 1:
                report = str('{phase} - Epoch: [{0}][{1}/{2}]\t'
                             'Time {meters[step].val:.3f} ({meters[step].avg:.3f})\t'
                             'Data {meters[data].val:.3f} ({meters[data].avg:.3f})\t'
                             'Loss {meters[loss].val:.4f} ({meters[loss].avg:.4f})\t'
                             'Prec@1 {meters[prec1].val:.3f} ({meters[prec1].avg:.3f})\t'
                             'Prec@5 {meters[prec5].val:.3f} ({meters[prec5].avg:.3f})\t'
                             # 'Samples {meters[samples].val}\t'
                             'Confidence {meters[confidence].val}\t'
                             .format(
                                 self.epoch, i, len(data_loader),
                                 phase='TRAINING' if training else 'EVALUATING',
                                 meters=meters))
                if 'grad' in meters.keys():
                    report += 'Grad {meters[grad].val:.3f} ({meters[grad].avg:.3f})'\
                        .format(meters=meters)
                logging.info(report)
                self.observe(trainer=self,
                             model=self._model,
                             optimizer=self.optimizer,
                             data=(inputs, target))
                self.stream_meters(meters,
                                   prefix='train' if training else 'eval')
                if training:
                    self.write_stream('lr',
                                      (self.training_steps, self.optimizer.get_lr()[0]))

            if num_steps is not None and i >= num_steps:
                break
        # if start_lr != self.optimizer.get_lr()[0] and start_lr != 0 and self.use_rand_selection == False:
        #     self.use_rand_selection = True
        #     print('Switching to random selection at epoch %d' % (self.epoch))
        # self.update_sel_ratio()

        return meter_results(meters)


    def train(self, data_loader, duplicates=1, chunk_batch=1):
        # switch to train mode
        self.model.train()

        return self.forward(data_loader, training=True, chunk_batch=chunk_batch)


    def validate(self, data_loader, duplicates=1):
        # switch to evaluate mode
        if self.training_steps % (self.train_batches * self.ratio) == 0:
            self.model.eval()
            with torch.no_grad():
                return super(SelectionTrainer, self).forward(data_loader, training=False)
        else:
            return None




class HardNegativeTrainer(Trainer):

    def __init__(self, *kargs,  **kwargs):
        self.ratio = kwargs.pop('ratio', 10)
        self.nm_min = kwargs.pop('nm_min', 0)
        self.nm_max = kwargs.pop('nm_max', 1)
        self.nm_rate = kwargs.pop('nm_rate', 1./10.)
        super(HardNegativeTrainer, self).__init__(*kargs,  **kwargs)



    def select_hard_samples(self, inputs, target, meters):
        self.model.eval()
        with torch.no_grad():

            logit, _ = self.model(inputs)

            scores = nn.functional.cross_entropy(logit, target, reduction='none')
            classes = target.unique()
            _, indices = scores.sort(dim=0, descending=True)
            target_ordered = target.index_select(
                0, indices.to(device=target.device))
            idx_target = list(zip(indices.tolist(), target_ordered.tolist()))
            classes = classes.tolist()
            shuffle(classes)
            classes_items = {c: [] for c in classes}
            ratio_hard = max(
                min(self.nm_max, (self.epoch + 1) * self.nm_rate), self.nm_min)

            def select_balanced(num, classes_items=classes_items, classes=classes, idx_target=idx_target):
                curr_c_num = 0
                while sum([len(c) for c in classes_items.values()]) < num:
                    for i, (idx, c) in enumerate(idx_target):
                        if c == classes[curr_c_num]:
                            idx_target.pop(i)
                            classes_items[c].append(idx)
                            break
                    curr_c_num += 1
                    if curr_c_num >= len(classes):
                        curr_c_num = 0

            select_balanced(ratio_hard * self.batch_size)
            shuffle(idx_target)
            select_balanced(self.batch_size)

            indices = []
            for v in classes_items.values():
                indices += v

            indices_left = list(set(range(inputs.size(0))) - set(indices))

            shuffle(indices)
            max_entr_indices1 = torch.tensor(indices, device=inputs.device, dtype=torch.long)
            # indices_left = torch.tensor(indices_left, device=inputs.device, dtype=torch.long)

            topk_vals, pred_classes = logit[max_entr_indices1].topk(2, 1)
            w1 = self.model.fc.weight.index_select(0, pred_classes[:, 0])
            w2 = self.model.fc.weight.index_select(0, pred_classes[:, 1])
            entropy_logit = (topk_vals[:,0] - topk_vals[:,1]) / (w1 - w2).norm(dim=-1)
            _, meas_indices = torch.sort(entropy_logit)

            confidence_top10 = torch.mean(entropy_logit[meas_indices][:10])
            meters['confidence'].update(float(confidence_top10), 1)

            # max_entr_indices1 = indices[:self.batch_size]
        self.model.train()
        return inputs[max_entr_indices1], target[max_entr_indices1] #, inputs[indices_left], target[indices_left]




    def forward(self, data_loader, num_steps=None, training=False, average_output=False, chunk_batch=1):
        self.train_batches = len(data_loader)
        meters = {name: AverageMeter()
                  for name in ['step', 'data', 'loss', 'prec1', 'prec5', 'samples', 'confidence']}
        if training and self.grad_clip > 0:
            meters['grad'] = AverageMeter()
        if self.calc_grad_var is not None:
            var_meter = OnlineMeter()
            meters['grad_var'] = AverageMeter()

        batch_first = True
        if training and isinstance(self.model, nn.DataParallel) or chunk_batch > 1:
            batch_first = False

        def meter_results(meters):
            results = {name: meter.avg for name, meter in meters.items()}
            results['error1'] = 100. - results['prec1']
            results['error5'] = 100. - results['prec5']
            return results

        end = time.time()


        for i, (inputs, target) in enumerate(data_loader):

            # measure data loading time
            meters['data'].update(time.time() - end)
            inputs = inputs.to(self.device, dtype=self.dtype)
            target = target.to(self.device)

            if training:
                
                inputs, target = self.select_hard_samples(inputs, target, meters)

              
            target = target.to(self.device)
            inputs = inputs.to(self.device, dtype=self.dtype)

            output, loss, grad = self._step(inputs, target,
                                            training=training,
                                            average_output=average_output,
                                            chunk_batch=chunk_batch)

            if self.calc_grad_var is not None:
                var_meter.update(self.collect_flatten_grads_(self.model.parameters()))
                if (self.training_steps + 1) % self.calc_grad_var == 0:
                    meters['grad_var'].update(float(var_meter.var.mean()), inputs.size(0))
                    var_meter.needs_init = True

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            meters['loss'].update(float(loss), inputs.size(0))
            meters['prec1'].update(float(prec1), inputs.size(0))
            meters['prec5'].update(float(prec5), inputs.size(0))
           
            if grad is not None:
                meters['grad'].update(float(grad), inputs.size(0))

            # measure elapsed time
            meters['step'].update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0 or i == len(data_loader) - 1:
                report = str('{phase} - Epoch: [{0}][{1}/{2}]\t'
                             'Time {meters[step].val:.3f} ({meters[step].avg:.3f})\t'
                             'Data {meters[data].val:.3f} ({meters[data].avg:.3f})\t'
                             'Loss {meters[loss].val:.4f} ({meters[loss].avg:.4f})\t'
                             'Prec@1 {meters[prec1].val:.3f} ({meters[prec1].avg:.3f})\t'
                             'Prec@5 {meters[prec5].val:.3f} ({meters[prec5].avg:.3f})\t'
                             # 'Samples {meters[samples].val}\t'
                             'Confidence {meters[confidence].val}\t'
                             .format(
                                 self.epoch, i, len(data_loader),
                                 phase='TRAINING' if training else 'EVALUATING',
                                 meters=meters))
                if 'grad' in meters.keys():
                    report += 'Grad {meters[grad].val:.3f} ({meters[grad].avg:.3f})'\
                        .format(meters=meters)
                logging.info(report)
                self.observe(trainer=self,
                             model=self._model,
                             optimizer=self.optimizer,
                             data=(inputs, target))
                self.stream_meters(meters,
                                   prefix='train' if training else 'eval')
                if training:
                    self.write_stream('lr',
                                      (self.training_steps, self.optimizer.get_lr()[0]))

            if num_steps is not None and i >= num_steps:
                break
       
        return meter_results(meters)


    def train(self, data_loader, duplicates=1, chunk_batch=1):
        # switch to train mode
        self.model.train()

        return self.forward(data_loader, training=True, chunk_batch=chunk_batch)


    def validate(self, data_loader, duplicates=1):
        # switch to evaluate mode
        if self.training_steps % (self.train_batches * self.ratio) == 0:
            self.model.eval()
            with torch.no_grad():
                return super(HardNegativeTrainer, self).forward(data_loader, training=False)
        else:
            return None


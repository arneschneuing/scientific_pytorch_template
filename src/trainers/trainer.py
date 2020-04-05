# Code adapted from:
# https://github.com/victoresque/pytorch-template/blob/master/base/
# base_trainer.py

import os
import torch
from abc import abstractmethod
from loggers.logger import Logger
from trainers.monitor import Monitor


class Trainer:
    """
    Trainer class
    :param result_dir:
    :param model:
    :param criterion:
    :param metric_tracker:
    :param optimizer:
    :param cfg:
    :param data_loaders:
    """
    def __init__(self, result_dir, model, criterion, metric_tracker, optimizer, cfg, data_loaders):
        self._cfg = cfg
        self._result_dir = result_dir
        os.mkdir(result_dir)
        self._logger = Logger(result_dir,
                              cfg['Logging'].get('write_file', default=True),
                              cfg['Logging'].get('write_tb', default=True))
        self._data_loaders = data_loaders

        # Use GPUs if available
        self._device, device_ids = self._prepare_device(cfg['n_gpu'])
        self._model = model.to(self._device)
        if len(device_ids) > 1:
            self._model = torch.nn.DataParallel(self._model,
                                                device_ids=device_ids)

        self._criterion = criterion
        self._metric_tracker = metric_tracker
        self._optimizer = optimizer

        self._num_iterations, self._checkpoint_freq, self._val_freq, \
            self.epoch_based = self._get_scheduling_params(cfg)

        patience = self._cfg['Monitor'].get('patience',
                                            default=self._num_iterations)
        self._monitor = Monitor(self._checkpoint_freq, patience)

        if cfg.resume is not None:
            self._resume_checkpoint(cfg.resume)

    def _get_scheduling_params(self, cfg):
        # Read from config file
        num_epochs = cfg['Train'].get('num_epochs', default=None)
        num_iterations = cfg['Train'].get('num_iterations', default=None)
        checkpoint_freq = cfg['Train'].get('checkpoint_freq', default=None)
        val_freq = cfg['Train'].get('val_freq', default=None)

        if num_epochs is None and num_iterations is None:
            self._logger.log_string('Error: please specify the training '
                                    'duration using num_epochs or '
                                    'num_iterations in config file')
            exit(-1)
        if num_epochs is not None and num_iterations is not None:
            self._logger.log_string('Error: training duration is ambiguous, '
                                    'num_epochs and num_iterations are '
                                    'specified in config file')
            exit(-1)

        if num_iterations is None:
            # epoch-based training
            num_iterations = len(self._data_loaders['train']) * num_epochs
            checkpoint_freq = num_iterations if checkpoint_freq is None \
                else len(self._data_loaders['train']) * checkpoint_freq
            val_freq = len(self._data_loaders['train']) if val_freq is None \
                else len(self._data_loaders['train']) * val_freq
            epoch_based = True
        else:
            # iteration-based training
            epoch_based = False

        return num_iterations, checkpoint_freq, val_freq, epoch_based

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _prepare_device(self, n_gpu):
        """
        Manage GPU setup
        :param n_gpu: intended number of GPUs
        """
        n_gpu_available = torch.cuda.device_count()
        if n_gpu_available == 0 and n_gpu > 0:
            self.logger.log_string("Warning: no GPU available, using CPU...")
            n_gpu = 0
        if n_gpu > n_gpu_available:
            self.logger.log_string(f"Warning: not enough GPUs available, "
                                   f"using {n_gpu_available} GPUs")
            n_gpu = n_gpu_available
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        list_ids = list(range(n_gpu))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'cfg': self.cfg
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['cfg']['arch'] != self.cfg['arch']:
            self.logger.warning("Warning: Architecture configuration given in cfg file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['cfg']['optimizer']['type'] != self.cfg['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in cfg file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
r"""
Early Stopping
==============

Monitor a validation metric and stop training when it stops improving.

"""
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist

from pytorch_lightning import _logger as log
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities import rank_zero_warn

torch_inf = torch.tensor(np.Inf)

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
except ImportError:
    XLA_AVAILABLE = False
else:
    XLA_AVAILABLE = True


class EarlyStopping(Callback):
    r"""

    Args:
        monitor: quantity to be monitored. Default: ``'val_loss'``.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than `min_delta`, will count as no
            improvement. Default: ``0``.
        patience: number of validation epochs with no improvement
            after which training will be stopped. Default: ``0``.
        verbose: verbosity mode. Default: ``False``.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity. Default: ``'auto'``.
        strict: whether to crash the training if `monitor` is
            not found in the validation metrics. Default: ``True``.

    Example::

        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.callbacks import EarlyStopping
        >>> early_stopping = EarlyStopping('val_loss')
        >>> trainer = Trainer(early_stop_callback=early_stopping)
    """
    mode_dict = {
        'min': torch.lt,
        'max': torch.gt,
    }

    def __init__(self, monitor: str = 'val_loss', min_delta: float = 0.0, patience: int = 3,
                 verbose: bool = False, mode: str = 'auto', strict: bool = True):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.strict = strict
        self.min_delta = min_delta
        self.wait_count = 0
        self.stopped_epoch = 0
        self.mode = mode

        if mode not in self.mode_dict:
            if self.verbose > 0:
                log.info(f'EarlyStopping mode {mode} is unknown, fallback to auto mode.')
            self.mode = 'auto'

        if self.mode == 'auto':
            if self.monitor == 'acc':
                self.mode = 'max'
            else:
                self.mode = 'min'
            if self.verbose > 0:
                log.info(f'EarlyStopping mode set to {self.mode} for monitoring {self.monitor}.')

        self.min_delta *= 1 if self.monitor_op == torch.gt else -1
        self.best_score = torch_inf if self.monitor_op == torch.lt else -torch_inf

    def _validate_condition_metric(self, logs):
        """
        Checks that the condition metric for early stopping is good

        Args:
            logs: callback metrics from validation output

        Return:
             True if specified metric is available
        """
        monitor_val = logs.get(self.monitor)
        error_msg = (f'Early stopping conditioned on metric `{self.monitor}`'
                     f' which is not available. Either add `{self.monitor}` to the return of '
                     f' validation_epoch end or modify your EarlyStopping callback to use any of the '
                     f'following: `{"`, `".join(list(logs.keys()))}`')

        if monitor_val is None:
            if self.strict:
                raise RuntimeError(error_msg)
            if self.verbose > 0:
                rank_zero_warn(error_msg, RuntimeWarning)

            return False

        return True

    @property
    def monitor_op(self):
        return self.mode_dict[self.mode]

    def state_dict(self):
        return {
            'wait_count': self.wait_count,
            'stopped_epoch': self.stopped_epoch,
            'best_score': self.best_score,
            'patience': self.patience
        }

    def load_state_dict(self, state_dict):
        state_dict = deepcopy(state_dict)
        self.wait_count = state_dict['wait_count']
        self.stopped_epoch = state_dict['stopped_epoch']
        self.best_score = state_dict['best_score']
        self.patience = state_dict['patience']

    def on_sanity_check_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        self._validate_condition_metric(logs)

    def on_validation_end(self, trainer, pl_module):
        self._run_early_stopping_check(trainer, pl_module)

    def _run_early_stopping_check(self, trainer, pl_module):
        logs = trainer.callback_metrics
        if not self._validate_condition_metric(logs):
            return  # short circuit if metric not present

        current = logs.get(self.monitor)
        if not isinstance(current, torch.Tensor):
            current = torch.tensor(current, device=pl_module.device)

        if trainer.use_tpu and XLA_AVAILABLE:
            current = current.cpu()

        if self.monitor_op(current - self.min_delta, self.best_score):
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            should_stop = self.wait_count >= self.patience

            if bool(should_stop):
                self.stopped_epoch = trainer.current_epoch
                trainer.should_stop = True

        # stop every ddp process if any world process decides to stop
        self._stop_distributed_training(trainer, pl_module)

    def _stop_distributed_training(self, trainer, pl_module):

        # in ddp make sure all processes stop when one is flagged
        if trainer.use_ddp or trainer.use_ddp2:
            stop = torch.tensor(int(trainer.should_stop), device=pl_module.device)
            dist.all_reduce(stop, op=dist.reduce_op.SUM)
            dist.barrier()
            trainer.should_stop = stop == trainer.world_size

        if trainer.use_tpu:
            stop = torch.tensor(int(trainer.should_stop), device=pl_module.device, dtype=torch.int32)
            stop = xm.mesh_reduce("stop_signal", stop, torch.cat)
            torch_xla.core.xla_model.rendezvous("pl.EarlyStoppingCallback.stop_distributed_training_check")
            trainer.should_stop = int(stop.item()) == trainer.world_size

    def on_train_end(self, trainer, pl_module):
        if self.stopped_epoch > 0 and self.verbose > 0:
            rank_zero_warn('Displayed epoch numbers by `EarlyStopping` start from "1" until v0.6.x,'
                           ' but will start from "0" in v0.8.0.', DeprecationWarning)
            log.info(f'Epoch {self.stopped_epoch + 1:05d}: early stopping triggered.')

"""
The lightning training loop handles everything except the actual computations of your model.
 To decide what will happen in your training loop, define the `training_step` function.

Below are all the things lightning automates for you in the training loop.

Accumulated gradients
---------------------

Accumulated gradients runs K small batches of size N before doing a backwards pass.
 The effect is a large effective batch size of size KxN.

.. code-block:: python

    # DEFAULT (ie: no accumulated grads)
    trainer = Trainer(accumulate_grad_batches=1)

Force training for min or max epochs
------------------------------------

It can be useful to force training for a minimum number of epochs or limit to a max number

.. code-block:: python

    # DEFAULT
    trainer = Trainer(min_epochs=1, max_epochs=1000)

Force disable early stop
------------------------

To disable early stopping pass None to the early_stop_callback

.. code-block:: python

    # DEFAULT
    trainer = Trainer(early_stop_callback=None)

Gradient Clipping
-----------------

Gradient clipping may be enabled to avoid exploding gradients.
 Specifically, this will `clip the gradient norm computed over all model parameters
 `together <https://pytorch.org/docs/stable/nn.html#torch.nn.utils.clip_grad_norm_>`_.

.. code-block:: python

    # DEFAULT (ie: don't clip)
    trainer = Trainer(gradient_clip_val=0)

    # clip gradients with norm above 0.5
    trainer = Trainer(gradient_clip_val=0.5)

Inspect gradient norms
----------------------

Looking at grad norms can help you figure out where training might be going wrong.

.. code-block:: python

    # DEFAULT (-1 doesn't track norms)
    trainer = Trainer(track_grad_norm=-1)

    # track the LP norm (P=2 here)
    trainer = Trainer(track_grad_norm=2)

Set how much of the training set to check
-----------------------------------------

If you don't want to check 100% of the training set (for debugging or if it's huge), set this flag.

limit_train_batches will be overwritten by overfit_batches if `overfit_batches > 0`

.. code-block:: python

    # DEFAULT
    trainer = Trainer(limit_train_batches=1.0)

    # check 10% only
    trainer = Trainer(limit_train_batches=0.1)

    # check 10 batches only
    trainer = Trainer(limit_train_batches=10)

Packed sequences as inputs
--------------------------

When using PackedSequence, do 2 things:
1. return either a padded tensor in dataset or a list of variable length tensors
in the dataloader collate_fn (example above shows the list implementation).
2. Pack the sequence in forward or training and validation steps depending on use case.

.. code-block:: python

    # For use in dataloader
    def collate_fn(batch):
        x = [item[0] for item in batch]
        y = [item[1] for item in batch]
        return x, y

    # In module
    def training_step(self, batch, batch_idx):
        x = rnn.pack_sequence(batch[0], enforce_sorted=False)
        y = rnn.pack_sequence(batch[1], enforce_sorted=False)


Truncated Backpropagation Through Time
--------------------------------------

There are times when multiple backwards passes are needed for each batch.
 For example, it may save memory to use Truncated Backpropagation Through Time when training RNNs.

When this flag is enabled each batch is split into sequences of size truncated_bptt_steps
 and passed to training_step(...) separately. A default splitting function is provided,
 however, you can override it for more flexibility. See `tbptt_split_batch`.

.. code-block:: python

    # DEFAULT (single backwards pass per batch)
    trainer = Trainer(truncated_bptt_steps=None)

    # (split batch into sequences of size 2)
    trainer = Trainer(truncated_bptt_steps=2)


NaN detection and intervention
------------------------------
When the `terminate_on_nan` flag is enabled, after every forward pass during training, Lightning will
check that

1. the loss you return in `training_step` is finite (not NaN and not +/-inf)
2. the model parameters have finite values.

Lightning will terminate the training loop with an error message if NaN or infinite
values are detected. If this happens, you should investigate numerically unstable operations
in your model.

.. code-block:: python

    # DEFAULT (won't perform the NaN check)
    trainer = Trainer(terminate_on_nan=False)

    # (NaN check each batch and terminate on NaN or infinite values)
    trainer = Trainer(terminate_on_nan=True)

"""

import subprocess
from abc import ABC, abstractmethod
from typing import Callable
from typing import Union, List

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.distributed as torch_distrib

from pytorch_lightning import _logger as log
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.trainer.supporters import TensorRunningAccum
from pytorch_lightning.utilities import rank_zero_warn, NATIVE_AMP_AVALAIBLE
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.parsing import AttributeDict
from pytorch_lightning.utilities.memory import recursive_detach
import os
from transformers import BartTokenizer
from gensim.corpora import Dictionary
from gensim.test.utils import datapath
from data_utils import DocDataset
import re
from mlutils.exp import yaml_load, yaml_dump
from mlutils.pt.training import GSMTrainer, extend_config_reference

try:
    from apex import amp
except ImportError:
    APEX_AVAILABLE = False
else:
    APEX_AVAILABLE = True

try:
    import torch_xla.distributed.parallel_loader as xla_pl
    import torch_xla.core.xla_model as xm
except ImportError:
    XLA_AVAILABLE = False
else:
    XLA_AVAILABLE = True

try:
    import horovod.torch as hvd
except (ModuleNotFoundError, ImportError):
    HOROVOD_AVAILABLE = False
else:
    HOROVOD_AVAILABLE = True

# constant which signals should be catched for graceful trainer shutdown
SIGNAL_TERMINATE = ('SIGTERM', 'SIGSEGV', 'SIGINT')


class TrainerTrainLoopMixin(ABC):
    # this is just a summary on variables used in this abstract class,
    #  the proper values/initialisation should be done in child class
    max_epochs: int
    min_epochs: int
    on_gpu: bool
    use_ddp: bool
    use_dp: bool
    use_ddp2: bool
    use_horovod: bool
    single_gpu: bool
    use_tpu: bool
    data_parallel_device_ids: ...
    check_val_every_n_epoch: ...
    num_training_batches: int
    val_check_batch: ...
    disable_validation: bool
    fast_dev_run: ...
    accumulation_scheduler: ...
    lr_schedulers: ...
    early_stop_callback: ...
    callback_metrics: ...
    logger: Union[LightningLoggerBase, bool]
    global_step: int
    testing: bool
    log_save_interval: float
    global_rank: int
    row_log_interval: float
    truncated_bptt_steps: ...
    optimizers: ...
    optimizer_frequencies: ...
    accumulate_grad_batches: int
    track_grad_norm: ...
    model: LightningModule
    interrupted: bool
    running_loss: ...
    progress_bar_dict: ...
    reduce_lr_on_plateau_scheduler: ...
    profiler: ...
    batch_idx: int
    precision: ...
    train_dataloader: DataLoader
    reload_dataloaders_every_epoch: bool
    max_steps: int
    min_steps: int
    total_batch_idx: int
    terminate_on_nan: bool
    tpu_id: int
    interactive_ddp_procs: ...

    # Callback system
    callbacks: List[Callback]
    on_train_start: Callable
    on_train_end: Callable
    on_batch_start: Callable
    on_batch_end: Callable
    on_epoch_start: Callable
    on_epoch_end: Callable
    on_validation_end: Callable
    on_keyboard_interrupt: Callable

    @abstractmethod
    def get_model(self) -> LightningModule:
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def is_function_implemented(self, *args, **kwargs):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def run_evaluation(self, *args, **kwargs):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def transfer_batch_to_gpu(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def transfer_batch_to_tpu(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def clip_gradients(self):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def detect_nan_tensors(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def is_overridden(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def add_progress_bar_metrics(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def log_metrics(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def process_output(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def reset_train_dataloader(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def reset_val_dataloader(self, model):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def has_arg(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    def train(self):
        # add signal handlers for process kills
        # def _signal_kill_handler(*args):
        #     return TrainerTrainLoopMixin.run_training_teardown(self)
        #
        # orig_signal_handlers = {}
        # for sig_name in SIGNAL_TERMINATE:
        #     orig_signal_handlers[sig_name] = signal.signal(getattr(signal, sig_name),
        #                                                    _signal_kill_handler)

        # get model
        model = self.get_model()

        # enable train mode
        model.train()

        # enable gradients
        torch.set_grad_enabled(True)

        # load data
        # if reload_dataloaders_every_epoch, this is moved to the epoch loop
        if not self.reload_dataloaders_every_epoch:
            self.reset_train_dataloader(model)
        self.reset_val_dataloader(model)

        # Train start events
        with self.profiler.profile('on_train_start'):
            # callbacks
            self.on_train_start()
            # model hooks
            model.on_train_start()

        try:
            # run all epochs
            for epoch in range(self.current_epoch, self.max_epochs):
                # reset train dataloader
                if self.reload_dataloaders_every_epoch:
                    self.reset_train_dataloader(model)
                # set seed for distributed sampler (enables shuffling for each epoch)
                if (self.use_ddp or self.use_horovod) \
                        and hasattr(self.train_dataloader, 'sampler') \
                        and hasattr(self.train_dataloader.sampler, 'set_epoch'):
                    self.train_dataloader.sampler.set_epoch(epoch)

                # update training progress in trainer and model
                model.current_epoch = epoch
                self.current_epoch = epoch

                # changing gradient according accumulation_scheduler
                self.accumulation_scheduler.on_epoch_start(self, self.get_model())

                # stores accumulated grad fractions per batch
                self.batch_loss_value = TensorRunningAccum(
                    window_length=self.accumulate_grad_batches
                )

                # -----------------
                # RUN TNG EPOCH
                # -----------------
                self.run_training_epoch()

                if self.max_steps and self.max_steps <= self.global_step:
                    self.run_training_teardown()
                    return

                # update LR schedulers
                self.update_learning_rates(interval='epoch')

                # early stopping
                met_min_epochs = epoch >= self.min_epochs - 1
                met_min_steps = self.global_step >= self.min_steps if self.min_steps else True

                if self.should_stop:
                    if (met_min_epochs and met_min_steps) or self.fast_dev_run:
                        self.run_training_teardown()
                        return
                    else:
                        log.info('Trainer was signaled to stop but required minimum epochs'
                                 f' ({self.min_epochs}) or minimum steps ({self.min_steps}) has'
                                 ' not been met. Training will continue...')

            self.run_training_teardown()

        except KeyboardInterrupt:
            rank_zero_warn('Detected KeyboardInterrupt, attempting graceful shutdown...')

            # user could press ctrl+c many times... only shutdown once
            if not self.interrupted:
                self.interrupted = True
                self.on_keyboard_interrupt()

                self.run_training_teardown()

    def prepare_train_loop_dataloader(self, train_dataloader):
        # on TPU we have to wrap it under the ParallelLoader
        if self.use_tpu:
            device = xm.xla_device(self.tpu_id)
            train_dataloader = xla_pl.ParallelLoader(train_dataloader, [device])
            train_dataloader = train_dataloader.per_device_loader(device)

        return train_dataloader

    def run_on_epoch_start_hook(self, model):
        # Epoch start events
        with self.profiler.profile('on_epoch_start'):
            # callbacks
            self.on_epoch_start()

            # model hooks
            if self.is_function_implemented('on_epoch_start'):
                model.on_epoch_start()

    def run_training_epoch(self):
        # get model
        model = self.get_model()

        # Epoch start events
        self.run_on_epoch_start_hook(model)

        # modify dataloader if needed (ddp, etc...)
        train_dataloader = self.prepare_train_loop_dataloader(self.train_dataloader)

        # bookkeeping
        epoch_output = []
        should_check_val = False

        # run epoch
        for batch_idx, (batch, is_last_batch) in self.profiler.profile_iterable(
                enumerate(_with_is_last(train_dataloader)), "get_train_batch"
        ):
            # stop epoch if we limited the number of training batches
            if batch_idx >= self.num_training_batches:
                break

            self.batch_idx = batch_idx
            model.global_step = self.global_step

            # ------------------------------------
            # TRAINING_STEP + TRAINING_STEP_END
            # ------------------------------------
            batch_output = self.run_training_batch(batch, batch_idx)

            # only track outputs when user implements training_epoch_end
            # otherwise we will build up unnecessary memory
            if self.is_overridden('training_epoch_end', model=self.get_model()):
                epoch_output.append(batch_output.training_step_output_for_epoch_end)

            # update LR schedulers
            self.update_train_loop_lr_schedulers()

            # when returning -1 from train_step, we end epoch early
            self.should_stop = batch_output.signal == -1

            # -----------------------------------------
            # VALIDATE IF NEEDED + CHECKPOINT CALLBACK
            # -----------------------------------------
            should_check_val = self.should_check_val(batch_idx, is_last_batch)
            if self.fast_dev_run or should_check_val:
                self.run_evaluation(test_mode=False)

            # -----------------------------------------
            # SAVE LOGGERS (ie: Tensorboard, etc...)
            # -----------------------------------------
            self.save_loggers_in_training_loop(batch_idx)

            # -----------------------------------------
            # SAVE METRICS TO LOGGERS
            # -----------------------------------------
            self.save_train_loop_metrics_to_loggers(batch_idx, batch_output)

            # progress global step according to grads progress
            self.increment_accumulated_grad_global_step()

            # max steps reached, end training
            if self.max_steps is not None and self.max_steps == self.global_step:
                break

            # end epoch early
            # stop when the flag is changed or we've gone past the amount
            # requested in the batches
            if self.fast_dev_run or self.should_stop:
                break

        # let ddp devices catch up when using horovod
        self.sync_horovod()

        # process epoch outputs
        self.run_training_epoch_end(epoch_output)

        # checkpoint callback
        self.check_checkpoint_callback(should_check_val)

        # epoch end hook
        self.run_on_epoch_end_hook(model)

    def check_checkpoint_callback(self, should_check_val):
        # when no val loop is present or fast-dev-run still need to call checkpoints
        # TODO bake this logic into the checkpoint callback
        should_activate = not self.is_overridden('validation_step') and not (self.fast_dev_run or should_check_val)
        if should_activate:
            checkpoint_callbacks = [c for c in self.callbacks if isinstance(c, ModelCheckpoint)]
            [c.on_validation_end(self, self.get_model()) for c in checkpoint_callbacks]

    def update_train_loop_lr_schedulers(self):
        if (self.batch_idx + 1) % self.accumulate_grad_batches == 0:
            # update lr
            self.update_learning_rates(interval='step')

    def run_on_epoch_end_hook(self, model):
        with self.profiler.profile('on_epoch_end'):
            # callbacks
            self.on_epoch_end()
            # model hooks
            if self.is_function_implemented('on_epoch_end'):
                model.on_epoch_end()

    def run_training_epoch_end(self, epoch_output):
        model = self.get_model()
        if self.is_overridden('training_epoch_end', model=model):
            self.global_step += 1
            epoch_output = model.training_epoch_end(epoch_output)
            _processed_outputs = self.process_output(epoch_output)
            log_epoch_metrics = _processed_outputs[2]
            callback_epoch_metrics = _processed_outputs[3]

            # add the metrics to the loggers
            self.log_metrics(log_epoch_metrics, {})

            # add metrics to callbacks
            self.callback_metrics.update(callback_epoch_metrics)

            # add metrics to progress_bar
            self.add_progress_bar_metrics(_processed_outputs[1])

    def sync_horovod(self):
        if self.use_horovod:
            hvd.join(hvd.local_rank() if self.on_gpu else -1)

    def increment_accumulated_grad_global_step(self):
        # progress global step according to grads progress
        if (self.batch_idx + 1) % self.accumulate_grad_batches == 0:
            self.global_step += 1
        self.total_batch_idx += 1

    def save_train_loop_metrics_to_loggers(self, batch_idx, batch_output):
        # when metrics should be logged
        should_log_metrics = batch_idx % self.row_log_interval == 0 or self.should_stop
        if should_log_metrics or self.fast_dev_run:
            # logs user requested information to logger
            self.log_metrics(batch_output.batch_log_metrics, batch_output.grad_norm_dic)

    def save_loggers_in_training_loop(self, batch_idx):
        # when loggers should save to disk
        should_save_log = (batch_idx + 1) % self.log_save_interval == 0 or self.should_stop
        if should_save_log or self.fast_dev_run:
            if self.is_global_zero and self.logger is not None:
                self.logger.save()

    def should_check_val(self, batch_idx, is_last_batch):
        # decide if we should run validation
        is_val_check_batch = (batch_idx + 1) % self.val_check_batch == 0
        can_check_epoch = (self.current_epoch + 1) % self.check_val_every_n_epoch == 0
        can_check_val = not self.disable_validation and can_check_epoch
        should_check_val = is_val_check_batch or self.should_stop
        is_last_batch_for_infinite_dataset = (is_last_batch and self.val_check_batch == float('inf'))
        should_check_val = can_check_val and (should_check_val or is_last_batch_for_infinite_dataset)

        return should_check_val

    def run_training_batch(self, batch, batch_idx):
        """

        :param batch: dict; contains three keys: input_ids, attention_mask, decoder_input_ids
            Example for 'batch':
                batch: {'input_ids': tensor([[  0,  36, 230,  ...,   8,  41,   2]]),
                'attention_mask': tensor([[1, 1, 1,  ..., 1, 1, 1]]),
                'decoder_input_ids': tensor([[    0,   287,    10,  2107,   111, 10468,   226, 47385, 11579,  1012,
                                                2156,     5,  5302, 47385,   281, 47385, 10003,   255, 47385,   347,
                                                111,  2107, 47385,   574, 47385,  1000, 47385,   398, 47385,   245,
                                                16,    10,   205,  1374, 12576,   479,   646,  1000,  1215,  3388,
                                                510,   742,    85,   128,   579,    65,     9,     5,   357,  3092,
                                                23,    63,  1836,    11,     5,  3555,   111,   672,  2156, 26180,
                                                47385,   642,   111,  3547,  4120,   479,   646,  1000,  1215,  3388,
                                                510,   742,  7192,  8806, 10262,  3444,  7951,  2170,  1318,     2]])}
        :param batch_idx: number of batch
        :return:
        """
        # load tokenizer
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        # load config for GSM
        config = yaml_load(f"{self.default_root_dir}/data/config/gsm.yaml")
        # load dict
        dictionary = Dictionary.load(datapath('dict-www-cnndm-unigram'))
        # remove [SEP]
        sep_list = ['[SEP_0]', '[SEP_1]', '[SEP_2]', '[SEP_3]', '[SEP_4]', '[SEP_5]', '[SEP_6]', '[SEP_7]',
                    '[SEP_8]', '[SEP_9]', '<S_SEP>']
        # vocab size for topic modeling
        vocab_size = len(dictionary)
        # model
        config['hidden']['features'][0] = vocab_size

        # trainer batch
        config['trainer_batch']['test_sample'] = 1
        config = extend_config_reference(config)
        gsm_trainer = config['GSMtrainer']
        gsm_trainer['base_dir'] = f"{self.default_root_dir}/log/bart-large-cnn-finetune"
        gsm_trainer = GSMTrainer.from_config(gsm_trainer)

        # number of topics
        K = config['gsmtopic']['k']

        # yaml_dump(gsm_trainer,
        #           os.path.join(f"{self.default_root_dir}/log/bart-large-cnn-finetune", "gsm_trainer.yaml"))

        # -----------------------------------------
        # Topic Modeling - GSM
        # -----------------------------------------
        batch_size = batch['input_ids'].size()[0]

        docs = []
        for batch_num in range(batch_size):
            # extract the batch_sentence
            batch_sentence = tokenizer.decode(batch['input_ids'][batch_num].tolist(), skip_special_tokens=True)
            # change to lowercase and split to list
            batch_sentence_list = batch_sentence.split(" ")
            # remove [SEP]
            batch_sentence_list_nosep = [item for item in batch_sentence_list if item not in sep_list]
            text = ' '.join([x for x in batch_sentence_list_nosep])
            fine_text = text.replace(' ##', '').lower()
            batch_sentence = re.sub(r'[^\w\s]', '', fine_text)
            # batch_sentence: change to the cleaned news for topic modeling
            # change to training data format in topic modeling
            gsm_data_bow = dictionary.doc2bow(batch_sentence.split(" "))
            docs.append(gsm_data_bow)

        # gsm_data: data for topic modeling
        gsm_data = DataLoader(DocDataset(docs, len(dictionary), device='cuda'), batch_size=config['dataset']['batch_size'], drop_last=False, num_workers=0)

        gsm_trainer.__dict__['train_iterator'] = gsm_data

        gsm_loss, gsm_p = gsm_trainer.co_train(vocab_size, training=True)

        del gsm_data

        # track grad norms
        grad_norm_dic = {}

        # track all metrics for callbacks
        batch_callback_metrics = []

        # track metrics to log
        batch_log_metrics = []

        if batch is None:
            return AttributeDict(signal=0, grad_norm_dic=grad_norm_dic)

        # Batch start events
        with self.profiler.profile('on_batch_start'):
            # callbacks
            self.on_batch_start()
            # hooks
            if self.is_function_implemented('on_batch_start'):
                response = self.get_model().on_batch_start(batch)
                if response == -1:
                    return AttributeDict(signal=-1, grad_norm_dic=grad_norm_dic)

        splits = [batch]
        if self.truncated_bptt_steps is not None:
            model_ref = self.get_model()
            with self.profiler.profile('tbptt_split_batch'):
                splits = model_ref.tbptt_split_batch(batch, self.truncated_bptt_steps)

        self.hiddens = None
        for split_idx, split_batch in enumerate(splits):
            self.split_idx = split_idx

            for opt_idx, optimizer in self._get_optimizers_iterable():
                # make sure only the gradients of the current optimizer's parameters are calculated
                # in the training step to prevent dangling gradients in multiple-optimizer setup.
                if len(self.optimizers) > 1:
                    for param in self.get_model().parameters():
                        param.requires_grad = False
                    for group in optimizer.param_groups:
                        for param in group['params']:
                            param.requires_grad = True

                # -------------------
                # calculate loss
                # -------------------
                beta = 0.01
                opt_closure_result = self.optimizer_closure(
                    split_batch,
                    batch_idx,
                    opt_idx,
                    optimizer,
                    self.hiddens,
                    gsm_p,  # topic distribution
                    gsm_loss,  # loss for topic modeling
                    K, # number of topics
                    beta,
                )

                # ------------------------------
                # POST forward bookkeeping
                # ------------------------------
                batch_callback_metrics.append(opt_closure_result.training_step_output.callback_metrics)
                batch_log_metrics.append(opt_closure_result.training_step_output.log_metrics)

                self.add_progress_bar_metrics(opt_closure_result.training_step_output.pbar_on_batch_end)

                # track hiddens
                self.hiddens = opt_closure_result.hiddens

                # check if loss or model weights are nan
                if self.terminate_on_nan:
                    self.detect_nan_tensors(opt_closure_result.loss)

                # track total loss for logging (avoid mem leaks)
                self.batch_loss_value.append(opt_closure_result.loss)

                # ------------------------------
                # BACKWARD PASS
                # ------------------------------
                # gradient update with accumulated gradients
                if (self.batch_idx + 1) % self.accumulate_grad_batches == 0:
                    # backward
                    grad_norm_dic = self.run_batch_backward_pass(split_batch, batch_idx, opt_idx, optimizer)

                    # calculate running loss for display
                    self.running_loss.append(self.batch_loss_value.mean())

                    # reset for next set of accumulated grads
                    self.batch_loss_value.reset()

        # Batch end events
        with self.profiler.profile('on_batch_end'):
            # callbacks
            self.on_batch_end()
            # model hooks
            if self.is_function_implemented('on_batch_end'):
                self.get_model().on_batch_end()

        # collapse all metrics into one dict
        batch_log_metrics = {k: v for d in batch_log_metrics for k, v in d.items()}

        # track all metrics for callbacks
        self.callback_metrics.update({k: v for d in batch_callback_metrics for k, v in d.items()})

        result = AttributeDict(
            signal=0,
            grad_norm_dic=grad_norm_dic,
            batch_log_metrics=batch_log_metrics,
            training_step_output_for_epoch_end=opt_closure_result.training_step_output_for_epoch_end
        )
        return result

    def run_batch_backward_pass(self, split_batch, batch_idx, opt_idx, optimizer):
        # ------------------
        # GRAD NORMS
        # ------------------
        # track gradient norms when requested
        grad_norm_dic = {}
        if batch_idx % self.row_log_interval == 0:
            if float(self.track_grad_norm) > 0:
                model = self.get_model()
                grad_norm_dic = model.grad_norm(
                    self.track_grad_norm)

        # ------------------
        # CLIP GRADS
        # ------------------
        if self.use_amp and NATIVE_AMP_AVALAIBLE and not self.use_tpu:
            self.scaler.unscale_(optimizer)
        self.clip_gradients()

        # ------------------
        # .STEP + ZERO_GRAD
        # ------------------
        self.call_optimizer_step(optimizer, opt_idx, batch_idx, split_batch)

        return grad_norm_dic

    def call_optimizer_step(self, optimizer, opt_idx, batch_idx, split_batch):
        # calls .step(), .zero_grad()
        # override function to modify this behavior
        model = self.get_model()

        with self.profiler.profile('optimizer_step'):
            lambda_closure = lambda: self.optimizer_closure(
                split_batch,
                batch_idx,
                opt_idx,
                optimizer,
                self.hiddens
            ).loss

            # apply TPU optimizer
            if self.use_tpu and XLA_AVAILABLE:
                model.optimizer_step(self.current_epoch, batch_idx,
                                     optimizer, opt_idx, lambda_closure, on_tpu=True)

            # for LBFGS do something a bit different
            elif isinstance(optimizer, torch.optim.LBFGS):

                # native amp + lbfgs is a no go right now
                if self.use_amp and NATIVE_AMP_AVALAIBLE:
                    raise MisconfigurationException(
                        'native PyTorch amp and lbfgs are not compatible.'
                        ' To request, please file a Github issue in PyTorch and tag @mcarilli')
                model.optimizer_step(self.current_epoch, batch_idx, optimizer, opt_idx, lambda_closure,
                                     using_lbfgs=True)

            # when using 16-bit
            else:
                native_amp = self.use_amp and NATIVE_AMP_AVALAIBLE
                model.optimizer_step(self.current_epoch, batch_idx, optimizer, opt_idx, lambda_closure,
                                     using_native_amp=native_amp)

            # in native 16-bit we need to update scaler after optimizer step
            if self.use_amp and NATIVE_AMP_AVALAIBLE and not self.use_tpu:
                self.scaler.update()

            # model hook
            model.on_before_zero_grad(optimizer)

            # clear gradients
            model.optimizer_zero_grad(self.current_epoch, batch_idx, optimizer, opt_idx)

    def optimizer_closure(self, split_batch, batch_idx, opt_idx, optimizer, hiddens, gsm_p, gsm_loss, K, beta=0.01):
        """
        wrap the forward step in a closure so second order methods work
        """
        # ---------------------------
        # FORWARD
        # ---------------------------
        with self.profiler.profile('model_forward'):
            if self.use_amp and NATIVE_AMP_AVALAIBLE and not self.use_tpu:
                with torch.cuda.amp.autocast():
                    training_step_output = self.training_forward(split_batch, batch_idx,
                                                                 opt_idx, hiddens, gsm_p, gsm_loss, K)
            else:
                training_step_output = self.training_forward(split_batch, batch_idx, opt_idx,
                                                             hiddens, gsm_p, gsm_loss, K)

            # ----------------------------
            # PROCESS THE RESULT
            # ----------------------------
            # format and reduce outputs accordingly
            training_step_output_for_epoch_end = training_step_output
            training_step_output = self.process_output(training_step_output, train=True)

            training_step_output = AttributeDict(
                batch_loss=training_step_output[0],
                pbar_on_batch_end=training_step_output[1],
                log_metrics=training_step_output[2],
                callback_metrics=training_step_output[3],
                hiddens=training_step_output[4],
            )

            # if the user decides to finally reduce things in epoch_end, save raw output without graphs
            training_step_output_for_epoch_end = recursive_detach(training_step_output_for_epoch_end)

        # accumulate loss
        # (if accumulate_grad_batches = 1 no effect)
        ## todo: check self.accumulate_grad_batches
        closure_loss = training_step_output.batch_loss / self.accumulate_grad_batches

        # ----------------------------
        # Calculate total loss
        # ----------------------------
        # closure_loss = (1 - beta) * closure_loss + beta * gsm_loss

        # the loss will get scaled for amp. avoid any modifications to it
        untouched_loss = closure_loss.detach().clone()

        # backward pass
        model_ref = self.get_model()
        with self.profiler.profile('model_backward'):
            # scale loss for 16 bit
            if self.precision == 16 and not self.on_tpu:
                closure_loss = model_ref.amp_scale_loss(closure_loss, optimizer, opt_idx)

                # enter amp context
                if not NATIVE_AMP_AVALAIBLE:
                    context = closure_loss
                    closure_loss = closure_loss.__enter__()

            # do backward pass
            model_ref.backward(self, closure_loss, optimizer, opt_idx)

            # exit amp context
            if self.precision == 16 and not NATIVE_AMP_AVALAIBLE and not self.on_tpu:
                a, b, c = None, None, None
                error = context.__exit__(a, b, c)
                if error:
                    rank_zero_warn(a, b, c)
                    raise Exception('apex unscale error')

            # once backward has been applied, release graph
            closure_loss = closure_loss.detach()
            training_step_output.batch_loss = training_step_output.batch_loss.detach()

        if self.use_horovod:
            # Synchronize Horovod to ensure gradient manipulations (e.g., loss scaling) are valid
            optimizer.synchronize()

        # insert after step hook
        if self.is_function_implemented('on_after_backward'):
            model_ref = self.get_model()
            with self.profiler.profile('on_after_backward'):
                model_ref.on_after_backward()

        result = AttributeDict(
            loss=untouched_loss,
            training_step_output=training_step_output,
            training_step_output_for_epoch_end=training_step_output_for_epoch_end,
            hiddens=training_step_output.hiddens,
        )
        return result

    def _get_optimizers_iterable(self):
        if not self.optimizer_frequencies:
            # call training_step once per optimizer
            return list(enumerate(self.optimizers))

        optimizer_freq_cumsum = np.cumsum(self.optimizer_frequencies)
        optimizers_loop_length = optimizer_freq_cumsum[-1]
        current_place_in_loop = self.total_batch_idx % optimizers_loop_length

        # find optimzier index by looking for the first {item > current_place} in the cumsum list
        opt_idx = np.argmax(optimizer_freq_cumsum > current_place_in_loop)
        return [(opt_idx, self.optimizers[opt_idx])]

    # @atexit.register
    def run_training_teardown(self):
        if hasattr(self, '_teardown_already_run') and self._teardown_already_run:
            return

        self._teardown_already_run = True

        # Train end events
        with self.profiler.profile('on_train_end'):
            # callbacks
            self.on_train_end()
            # model hooks
            if self.is_function_implemented('on_train_end'):
                self.get_model().on_train_end()

        if self.logger is not None:
            self.logger.finalize("success")

        # summarize profile results
        if self.global_rank == 0:
            self.profiler.describe()

        if self.global_rank == 0:
            for proc in self.interactive_ddp_procs:
                subprocess.Popen.kill(proc)

        # clean up dist group
        if self.use_ddp or self.use_ddp2:
            torch_distrib.destroy_process_group()

        # clear mem
        if self.on_gpu:
            model = self.get_model()
            model.cpu()
            torch.cuda.empty_cache()

    def training_forward(self, batch, batch_idx, opt_idx, hiddens, gsm_p, gsm_loss, K):
        """
        Handle forward for each training case (distributed, single gpu, etc...)
        :param batch: single batch dict
        :param batch_idx:
        :return:
        """
        # ---------------
        # FORWARD
        # ---------------
        # enable not needing to add opt_idx to training_step
        args = [batch, batch_idx]

        if len(self.optimizers) > 1:
            if self.has_arg('training_step', 'optimizer_idx'):
                args.append(opt_idx)
            else:
                num_opts = len(self.optimizers)
                raise ValueError(
                    f'Your LightningModule defines {num_opts} optimizers but '
                    f'training_step is missing the "optimizer_idx" argument.'
                )

        # pass hiddens if using tbptt
        if self.truncated_bptt_steps is not None:
            args.append(hiddens)

        # distributed forward
        if self.use_ddp or self.use_ddp2 or self.use_dp:
            # self.model: LightningDistributedDataParallel(
            #   (module): SummarizationModule(
            #     (model): BartForConditionalGeneration(
            #       (model): BartModel(
            #         (shared): Embedding(50264, 1024, padding_idx=1)
            #         (encoder): BartEncoder()...

            args[0]['topic_p'] = gsm_p
            # args[0]['tm_loss'] = gsm_loss
            # args[0]['K'] = K
            # self.model => SummarizationModule
            output = self.model(*args)

        # Horovod
        elif self.use_horovod and self.on_gpu:
            batch = self.transfer_batch_to_gpu(batch, hvd.local_rank())
            args[0] = batch
            output = self.model.training_step(*args)

        # single GPU forward
        elif self.single_gpu:
            gpu_id = 0
            if isinstance(self.data_parallel_device_ids, list):
                gpu_id = self.data_parallel_device_ids[0]

            # Don't copy the batch since there is a single gpu that the batch could
            # be referenced from and if there are multiple optimizers the batch will
            # wind up copying it to the same device repeatedly.
            batch = self.transfer_batch_to_gpu(batch, gpu_id)
            args[0] = batch
            output = self.model.training_step(*args)

        # TPU support
        elif self.use_tpu:
            batch = self.transfer_batch_to_tpu(batch, self.tpu_id)
            args[0] = batch
            output = self.model.training_step(*args)

        # CPU forward
        else:
            output = self.model.training_step(*args)

        # allow any mode to define training_step_end
        # do something will all the dp outputs (like softmax)
        if self.is_overridden('training_step_end'):
            model_ref = self.get_model()
            with self.profiler.profile('training_step_end'):
                output = model_ref.training_step_end(output)

        # allow any mode to define training_end
        if self.is_overridden('training_end'):
            model_ref = self.get_model()
            with self.profiler.profile('training_end'):
                output = model_ref.training_end(output)

        return output

    def update_learning_rates(self, interval: str):
        """Update learning rates.

        Args:
            interval: either 'epoch' or 'step'.
        """
        if not self.lr_schedulers:
            return

        for lr_scheduler in self.lr_schedulers:
            current_idx = self.batch_idx if interval == 'step' else self.current_epoch
            current_idx += 1  # account for both batch and epoch starts from 0
            # Take step if call to update_learning_rates matches the interval key and
            # the current step modulo the schedulers frequency is zero
            if lr_scheduler['interval'] == interval and current_idx % lr_scheduler['frequency'] == 0:
                # If instance of ReduceLROnPlateau, we need to pass validation loss
                if lr_scheduler['reduce_on_plateau']:
                    monitor_key = lr_scheduler['monitor']
                    monitor_val = self.callback_metrics.get(monitor_key)
                    if monitor_val is None:
                        avail_metrics = ','.join(list(self.callback_metrics.keys()))
                        raise MisconfigurationException(
                            f'ReduceLROnPlateau conditioned on metric {monitor_key}'
                            f' which is not available. Available metrics are: {avail_metrics}.'
                            ' Condition can be set using `monitor` key in lr scheduler dict'
                        )
                    lr_scheduler['scheduler'].step(monitor_val)
                else:
                    lr_scheduler['scheduler'].step()


def _with_is_last(iterable):
    """Pass through values from the given iterable with an added boolean indicating if this is the last item.
    See `https://stackoverflow.com/a/1630350 <https://stackoverflow.com/a/1630350>`_"""
    it = iter(iterable)
    last = next(it)
    for val in it:
        # yield last and has next
        yield last, False
        last = val
    # yield last, no longer has next
    yield last, True

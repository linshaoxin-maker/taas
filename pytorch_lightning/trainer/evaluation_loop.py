"""
Validation loop
===============

The lightning validation loop handles everything except the actual computations of your model.
To decide what will happen in your validation loop, define the `validation_step` function.
Below are all the things lightning automates for you in the validation loop.

.. note:: Lightning will run 5 steps of validation in the beginning of training as a sanity
 check so you don't have to wait until a full epoch to catch possible validation issues.

Check validation every n epochs
-------------------------------

If you have a small dataset you might want to check validation every n epochs

.. code-block:: python

    # DEFAULT
    trainer = Trainer(check_val_every_n_epoch=1)

Set how much of the validation set to check
-------------------------------------------

If you don't want to check 100% of the validation set (for debugging or if it's huge), set this flag.

limit_val_batches will be overwritten by overfit_batches if `overfit_batches > 0`

.. code-block:: python

    # DEFAULT
    trainer = Trainer(limit_val_batches=1.0)

    # check 10% only
    trainer = Trainer(limit_val_batches=0.1)

Set how much of the test set to check
-------------------------------------

If you don't want to check 100% of the test set (for debugging or if it's huge), set this flag.

limit_test_batches will be overwritten by overfit_batches if `overfit_batches > 0`

.. code-block:: python

    # DEFAULT
    trainer = Trainer(limit_test_batches=1.0)

    # check 10% only
    trainer = Trainer(limit_test_batches=0.1)

Set validation check frequency within 1 training epoch
------------------------------------------------------

For large datasets it's often desirable to check validation multiple times within a training loop.
 Pass in a float to check that often within 1 training epoch.
 Pass in an int k to check every k training batches. Must use an int if using an IterableDataset.

.. code-block:: python

    # DEFAULT
    trainer = Trainer(val_check_interval=0.95)

    # check every .25 of an epoch
    trainer = Trainer(val_check_interval=0.25)

    # check every 100 train batches (ie: for IterableDatasets or fixed frequency)
    trainer = Trainer(val_check_interval=100)


Set the number of validation sanity steps
-----------------------------------------

Lightning runs a few steps of validation in the beginning of training.
 This avoids crashing in the validation loop sometime deep into a lengthy training loop.

.. code-block:: python

    # DEFAULT
    trainer = Trainer(num_sanity_val_steps=5)


You can use `Trainer(num_sanity_val_steps=0)` to skip the sanity check.

# Testing loop

To ensure you don't accidentally use test data to guide training decisions Lightning
 makes running the test set deliberate.

**test**

You have two options to run the test set.
First case is where you test right after a full training routine.

.. code-block:: python

    # run full training
    trainer.fit(model)

    # run test set
    trainer.test()


Second case is where you load a model and run the test set

.. code-block:: python

    model = MyLightningModule.load_from_checkpoint(
        checkpoint_path='/path/to/pytorch_checkpoint.ckpt',
        hparams_file='/path/to/test_tube/experiment/version/hparams.yaml',
        map_location=None
    )

    # init trainer with whatever options
    trainer = Trainer(...)

    # test (pass in the model)
    trainer.test(model)

In this second case, the options you pass to trainer will be used when running
 the test set (ie: 16-bit, dp, ddp, etc...)

"""

from abc import ABC, abstractmethod
from pprint import pprint
from typing import Callable, List, Union

import torch
from torch.utils.data import DataLoader

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel, LightningDataParallel
from pytorch_lightning.utilities import rank_zero_warn, NATIVE_AMP_AVALAIBLE
from torch import distributed as dist
from transformers import BartTokenizer
from gensim.corpora import Dictionary
from gensim.test.utils import datapath
from data_utils import DocDataset
import re
import os
from torch.autograd import Variable
from mlutils.exp import yaml_load
from mlutils.pt.training import GSMTrainer, extend_config_reference

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


class TrainerEvaluationLoopMixin(ABC):
    # this is just a summary on variables used in this abstract class,
    #  the proper values/initialisation should be done in child class
    on_gpu: bool
    use_ddp: bool
    use_dp: bool
    use_ddp2: bool
    use_horovod: bool
    single_gpu: bool
    data_parallel_device_ids: ...
    model: LightningModule
    num_test_batches: List[int]
    num_val_batches: int
    world_size: int
    fast_dev_run: ...
    process_output: ...
    progress_bar_dict: ...
    global_rank: int
    current_epoch: int
    callback_metrics: ...
    test_dataloaders: DataLoader
    val_dataloaders: DataLoader
    use_tpu: bool
    reload_dataloaders_every_epoch: ...
    tpu_id: int

    # Callback system
    on_validation_batch_start: Callable
    on_validation_batch_end: Callable
    on_test_batch_start: Callable
    on_test_batch_end: Callable
    on_validation_start: Callable
    on_validation_end: Callable
    on_test_start: Callable
    on_test_end: Callable

    @abstractmethod
    def copy_trainer_model_properties(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def get_model(self) -> LightningModule:
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def is_overridden(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def transfer_batch_to_tpu(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def transfer_batch_to_gpu(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def add_progress_bar_metrics(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def log_metrics(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def reset_test_dataloader(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def reset_val_dataloader(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    def _evaluate(
            self,
            model: LightningModule,
            dataloaders: List[DataLoader],
            max_batches: Union[int, List[int]],
            test_mode: bool = False
    ):
        """Run evaluation code.
        Args:
            model: The model to evaluate.
            dataloaders: A list of PyTorch dataloaders.
            max_batches: An integer or list of integers with length of the number of dataloaders. Each
                entry is the number of batches to process in the corresponding dataloader.
            test_mode:
        """
        # enable eval mode
        model.zero_grad()
        model.eval()

        # copy properties for forward overrides
        self.copy_trainer_model_properties(model)

        # bookkeeping
        outputs = []

        # convert max_batches to list
        if isinstance(max_batches, int):
            max_batches = [max_batches] * len(dataloaders)

        # run validation
        for dataloader_idx, dataloader in enumerate(dataloaders):
            dl_outputs = []

            # on TPU we have to wrap it under the ParallelLoader
            if self.use_tpu:
                device = xm.xla_device(self.tpu_id)
                dataloader = xla_pl.ParallelLoader(dataloader, [device])
                dataloader = dataloader.per_device_loader(device)

            # each dataloader has a max num batches
            dl_max_batches = max_batches[dataloader_idx]

            # load tokenizer
            tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

            # load dict
            dictionary = Dictionary.load(datapath('dict-www-cnndm-unigram'))

            # vocab size for topic modeling
            vocab_size = len(dictionary)

            for batch_idx, batch in enumerate(dataloader):
                if batch is None:
                    continue

                # stop short when on fast_dev_run (sets max_batch=1)
                if batch_idx >= dl_max_batches:
                    break

                # -----------------
                # Topic Modeling
                # -----------------
                # load config for GSM
                config = yaml_load(f"{self.default_root_dir}/data/config/gsm.yaml")

                # remove [SEP]
                sep_list = ['[SEP_0]', '[SEP_1]', '[SEP_2]', '[SEP_3]', '[SEP_4]', '[SEP_5]', '[SEP_6]', '[SEP_7]',
                            '[SEP_8]', '[SEP_9]']

                # model
                config['hidden']['features'][0] = vocab_size

                # trainer batch
                config['trainer_batch']['test_sample'] = 1
                config = extend_config_reference(config)
                gsm_trainer = config['GSMtrainer']
                gsm_trainer['base_dir'] = f"{self.default_root_dir}/log/bart-large-cnn-finetune"
                gsm_trainer = GSMTrainer.from_config(gsm_trainer)

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

                gsm_loss, gsm_p = gsm_trainer.co_train(vocab_size, training=False)

                del gsm_data

                topic_p = Variable(gsm_p.data, requires_grad=False)
                # tm_loss = Variable(gsm_loss.data, requires_grad=False)

                # disable gradients to save memory
                torch.set_grad_enabled(False)

                # callbacks
                if test_mode:
                    self.on_test_batch_start()
                else:
                    self.on_validation_batch_start()

                # -----------------
                # RUN EVALUATION STEP
                # -----------------
                beta = 0.01
                if self.use_amp and NATIVE_AMP_AVALAIBLE and not self.use_tpu:
                    with torch.cuda.amp.autocast():
                        output = self.evaluation_forward(model, batch, batch_idx, dataloader_idx, topic_p, gsm_loss, beta, test_mode)
                else:
                    output = self.evaluation_forward(model, batch, batch_idx, dataloader_idx, topic_p, gsm_loss, beta, test_mode)

                # on dp / ddp2 might still want to do something with the batch parts
                if test_mode:
                    if self.is_overridden('test_step_end'):
                        model_ref = self.get_model()
                        with self.profiler.profile('test_step_end'):
                            output = model_ref.test_step_end(output)
                    self.on_test_batch_end()
                else:
                    if self.is_overridden('validation_step_end'):
                        model_ref = self.get_model()
                        with self.profiler.profile('validation_step_end'):
                            output = model_ref.validation_step_end(output)
                    self.on_validation_batch_end()

                # track outputs for collation
                dl_outputs.append(output)

                # enable gradients to save memory
                torch.set_grad_enabled(True)

            outputs.append(dl_outputs)

        eval_results = {}

        # with a single dataloader don't pass an array
        if len(dataloaders) == 1:
            outputs = outputs[0]

        # give model a chance to do something with the outputs (and method defined)
        if isinstance(model, (LightningDistributedDataParallel, LightningDataParallel)):
            model = model.module

        if test_mode:
            if self.is_overridden('test_end', model=model):
                eval_results = model.test_end(outputs)
                rank_zero_warn('Method `test_end` was deprecated in v0.7 and will be removed in v1.0.'
                               ' Use `test_epoch_end` instead.', DeprecationWarning)

            elif self.is_overridden('test_epoch_end', model=model):
                eval_results = model.test_epoch_end(outputs)

        else:
            if self.is_overridden('validation_end', model=model):
                eval_results = model.validation_end(outputs)
                rank_zero_warn('Method `validation_end` was deprecated in v0.7 and will be removed in v1.0.'
                               ' Use `validation_epoch_end` instead.', DeprecationWarning)

            elif self.is_overridden('validation_epoch_end', model=model):
                eval_results = model.validation_epoch_end(outputs)

        # enable train mode again
        model.train()

        return eval_results


    def run_evaluation(self, test_mode: bool = False):
        # hook
        model = self.get_model()
        model.on_pre_performance_check()

        # select dataloaders
        if test_mode:
            self.reset_test_dataloader(model)

            dataloaders = self.test_dataloaders
            max_batches = self.num_test_batches
        else:
            # val
            if self.val_dataloaders is None:
                self.reset_val_dataloader(model)

            dataloaders = self.val_dataloaders
            max_batches = self.num_val_batches

        # enable fast_dev_run without val loop
        if dataloaders is None:
            return

        # cap max batches to 1 when using fast_dev_run
        if self.fast_dev_run:
            max_batches = [1]

        # Validation/Test begin callbacks
        if test_mode:
            self.on_test_start()
        else:
            self.on_validation_start()

        # enable disabling validation step with limit_val_batches = 0
        should_skip = sum(max_batches) == 0
        if should_skip:
            return

        # run evaluation
        eval_results = self._evaluate(self.model, dataloaders, max_batches, test_mode)

        # enable no returns
        callback_metrics = {}
        if eval_results is not None and len(eval_results) > 0:
            _, prog_bar_metrics, log_metrics, callback_metrics, _ = self.process_output(eval_results)

            # add metrics to prog bar
            self.add_progress_bar_metrics(prog_bar_metrics)

            # log results of test
            if test_mode and self.is_global_zero:
                print('-' * 80)
                print('TEST RESULTS')
                pprint(callback_metrics)
                print('-' * 80)

            # log metrics
            self.log_metrics(log_metrics, {})

            # track metrics for callbacks
            self.callback_metrics.update(callback_metrics)

        # hook
        model.on_post_performance_check()

        # eventual dataset reloading
        if test_mode:
            if self.reload_dataloaders_every_epoch:
                self.reset_test_dataloader(model)
        else:
            # val
            if self.reload_dataloaders_every_epoch:
                self.reset_val_dataloader(model)

        # Validation/Test end callbacks
        if test_mode:
            self.on_test_end()
        else:
            self.on_validation_end()

        return callback_metrics


    def evaluation_forward(self, model, batch, batch_idx, dataloader_idx, topic_p, topic_loss, K, beta, test_mode: bool = False):
        # make dataloader_idx arg in validation_step optional
        args = [batch, batch_idx]

        if (test_mode and len(self.test_dataloaders) > 1) \
                or (not test_mode and len(self.val_dataloaders) > 1):
            args.append(dataloader_idx)

        # handle DP, DDP forward
        if self.use_ddp or self.use_dp or self.use_ddp2:
            args[0]['topic_p'] = topic_p
            # args[0]['tm_loss'] = topic_loss
            # args[0]['K'] = K
            # args[0]['beta'] = beta
            output = model(*args)
            return output

        # Horovod
        if self.use_horovod and self.on_gpu:
            batch = self.transfer_batch_to_gpu(batch, hvd.local_rank())
            args[0] = batch

        # single GPU data transfer
        if self.single_gpu:
            # for single GPU put inputs on gpu manually
            root_gpu = 0
            if isinstance(self.data_parallel_device_ids, list):
                root_gpu = self.data_parallel_device_ids[0]
            batch = self.transfer_batch_to_gpu(batch, root_gpu)
            args[0] = batch

        # TPU data  transfer
        if self.use_tpu:
            batch = self.transfer_batch_to_tpu(batch, self.tpu_id)
            args[0] = batch

        # CPU, TPU or gpu step
        if test_mode:
            output = model.test_step(*args)
        else:
            output = model.validation_step(*args)

        return output

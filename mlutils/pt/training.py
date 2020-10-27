import torch
from torch.nn.utils import clip_grad_value_
import os
from os import path
import math
import sys
import re
import numpy as np
import json
from importlib import import_module
import six
import logging
import copy
from tensorboardX import SummaryWriter
from sklearn.metrics import classification_report
from scipy.special import softmax
from scipy.sparse.linalg import svds
from sklearn.decomposition import NMF
import torch.nn.functional as F
from torch import nn

from ..callbacks import CallbackList


def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


class RunningStatistics:

    def __init__(self):
        self.statistics = None

    def add(self, stat):
        """将每个batch获得的统计数据进行累积
        stat: 字典，key为对应的统计量 value 是一个scalar 或者 字典包含value和weight
        """
        if self.statistics is None:
            self.statistics = {}
            for k, v in stat.items():
                if isinstance(v, dict):
                    assert 'value' in v and 'weight' in v, 'value weight should be in {}'.format(str(v))
                    self.statistics[k] = v
                else:
                    self.statistics[k] = {'value': v, 'weight': 1}
        else:
            for k, v in stat.items():
                if not isinstance(v, dict):
                    v = {'value': v, 'weight': 1}
                last_v = self.statistics[k]['value']
                last_w = self.statistics[k]['weight']
                self.statistics[k]['value'] = (v['value'] * v['weight'] + last_v * last_w) / (last_w + v['weight'])
                self.statistics[k]['weight'] += v['weight']
        return self

    def description(self, prefix=''):
        """转化成字符串供输出"""
        if self.statistics is None:
            return 'None'
        return ' | '.join(['{} v{:.6f} w{:.6f}'.format(prefix + k, v['value'], v['weight'])
                           for k, v in self.statistics.items()])

    def get_value(self, k):
        """得到对应k的累值"""
        return self.statistics[k]['value']

    def get_dict(self):
        if self.statistics is None:
            return {}
        return {k: v['value'] for k, v in self.statistics.items()}


def eval_metric_cmp_key(key='loss', cmp=np.less):
    """Return true if new is better than old.
    key: loss, cmp: np.less
    key: accuracy cmp: np.greater
    """
    return lambda new, old: cmp(new.get_value(key), old.get_value(key))



###################################################
# metrics
###################################################


def classification_accuracy(pred, gold_true):
    """pred, gold_ture: torch tensors or numpy tensors."""
    if isinstance(pred, torch.Tensor):
        arg_max = torch.argmax(pred, dim=-1, keepdim=False)
        return (arg_max == gold_true).float().mean().item()
    return (np.argmax(pred, axis=-1) == gold_true).mean()


###################################################
# config ralated
###################################################


def parse_class(dotted_path: str):
    """
    Import a dotted module path and return the attribute/class designated by the
    last name in the path. Raise ImportError if the import failed.
    """
    try:
        module_path, class_name = dotted_path.rsplit('.', 1)
    except ValueError:
        msg = "%s doesn't look like a module path" % dotted_path
        six.reraise(ImportError, ImportError(msg), sys.exc_info()[2])

    module = import_module(module_path)

    try:
        return getattr(module, class_name)
    except AttributeError:
        msg = 'Module "%s" does not define a "%s" attribute/class' % (
            module_path, class_name)
        six.reraise(ImportError, ImportError(msg), sys.exc_info()[2])


def parse_config_type(config):
    """返回config指定的类或函数对象"""
    if 'type' in config:
        return parse_class(config['type'])
    return eval(config['eval'])


def build_from_config(config):
    """build classes or functions specified in config.
    config file format:
    1. callable
    {
    type: classTypeStr(or function)
    args: ...
    }
    2. just eval
    {
    eval: classTypeStr
    }  -> eval(classTypeStr): useful when class name or function name is needed
    """
    if 'type' not in config:
        assert 'eval' in config, 'eval not in config {}'.format(str(config))
        return eval(config['eval'])
    for k in config.keys():
        if isinstance(config[k], dict):
            config[k] = build_from_config(config[k])
    return parse_class(config.pop('type'))(**config)


def extend_config_reference(config):
    """Extend the reference in config. Make sure that no circular reference in config.
    config:
    {'a': 'b',
    'b': {}} ->
    {'a': {}(which is denoted by 'b'),
    'b': {}}
    """

    def _parse_reference(keys, r):
        if hasattr(r, '__getitem__'):
            try:
                v = r.__getitem__(keys)
                return v
            except (KeyError, TypeError, IndexError):
                pass
        if isinstance(keys, tuple):
            v = _parse_reference(keys[0], r)
            if v is not None:
                if len(keys) == 1:
                    return v
                return _parse_reference(keys[1:], v)
        return None

    def _sub_reference(cf, ori):
        it = cf.keys() if isinstance(cf, dict) else range(len(cf))
        for k in it:
            v = cf[k]
            if isinstance(v, (dict, list)):
                v = _sub_reference(v, ori)
            else:
                r = _parse_reference(v, ori)
                if r is not None:
                    v = r
            cf[k] = v
        return cf

    replace = copy.deepcopy(config)
    return _sub_reference(replace, replace)


class TrainerBatch:
    def __init__(self,
                 model,
                 optimizer,
                 criterion,
                 data_attrs,  # List of attributions of a batch of data. The order is important.
                 label_attr,
                 device,
                 grad_clip_value=0,
                 ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.data_attrs = data_attrs if isinstance(data_attrs, list) else [data_attrs]
        self.label_attr = label_attr
        self.dst_device = device
        self.grad_clip_value = grad_clip_value

    def place_data_to_device(self, x):
        if x.device != 'cuda':
            return x.to('cuda')
        return x

    def attrs_from_batch(self, batch, attr):
        single = False
        if not isinstance(attr, list):
            single = True
            attr = [attr]
        if isinstance(batch, list):
            batch = [self.place_data_to_device(b) for b in batch]
        elif isinstance(batch, dict):
            batch = [self.place_data_to_device(batch[d]) for d in attr]
        else:
            batch = [self.place_data_to_device(getattr(batch, d)) for d in attr]
        return batch if not single else batch[0]

    def train_batch(self, data, train=True):
        self.model.train(train)

    def eval_batch(self, data):
        return self.train_batch(data, train=False)

    def predict_batch(self, data, prob):
        """if not prob, return the argmax with dim = -1"""
        self.model.eval()
        with torch.no_grad():
            x = self.attrs_from_batch(data, self.data_attrs)
            out = self.model(*x)
        if not prob:
            return torch.argmax(out, dim=-1)
        return torch.softmax(out, dim=-1)

    @staticmethod
    def from_config(config):
        config['model'] = build_from_config(config['model']).to(config['device'])
        if 'type' in config['optimizer']:
            optimizer = parse_class(config['optimizer'].pop('type'))
            optimizer = optimizer(config['model'].parameters(), **config['optimizer'])
        else:
            optimizer = eval(config['optimizer']['eval'])
        config['optimizer'] = optimizer
        return build_from_config(config)


class GSMTrainer:
    def __init__(self,
                 base_dir,
                 num_epochs,
                 trainer_batch,
                 train_iterator=None,
                 dev_iterator=None,
                 test_iterator=None,
                 test_report=classification_report,  # report about (y_true, y_pred)
                 eval_metric=eval_metric_cmp_key(),  # True if (New, Old) -> True
                 statistics=RunningStatistics,
                 early_stop=None,
                 evaluate_interval=None,
                 save_checkpoint_interval=None,
                 num_checkpoints_keep=10,
                 print_statistics_interval=None,
                 callbacks=None,
                 logger=None):
        os.makedirs(base_dir, exist_ok=True)
        self.base_dir = base_dir
        if logger is None:
            logger = get_logger(path.join(base_dir, 'log'))
        self.logger = logger
        self.train_iterator = train_iterator
        self.dev_iterator = dev_iterator
        self.test_iterator = test_iterator
        self.test_report = test_report
        self.num_epochs = num_epochs
        self.trainer_batch = trainer_batch
        self.save_checkpoint_interval = math.inf if save_checkpoint_interval is None else save_checkpoint_interval
        # self.print_statistics_interval = len(train_iterator) if print_statistics_interval is None else print_statistics_interval
        # self.evaluate_interval = len(train_iterator) if evaluate_interval is None else evaluate_interval
        self.eval_metric = eval_metric
        self.best_eval = None
        self.early_stop = early_stop
        self._early_stop_counter = 0
        self.statistics = statistics
        self.num_checkpoints_keep = num_checkpoints_keep
        self.checkpoints = []
        # self.summary_writer = SummaryWriter(path.join(base_dir, 'summary'))

        callbacks = callbacks or []
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        self.callbacks = CallbackList(callbacks)
        self.callbacks.set_trainer(self)

    def _evaluate_epoch(self):
        stat = self.statistics()
        for batch in self.dev_iterator:
            stat.add(self.trainer_batch.eval_batch(batch))
        return stat

    def co_train(self, vocab_size, training):
        # res: contain loss, elbo etc. information
        # topic: topic distribution over words; size: torch.Size([50, 1994])
        for b, batch in enumerate(self.train_iterator):
            # b: int
            # res: dict
            res, topic = self.trainer_batch.train_batch(batch, train=training)

        # using ffn to reduce the dimension
        fc1 = nn.Linear(vocab_size, 1024).cuda()
        fc2 = nn.LayerNorm(1024).cuda()

        residual = fc1(topic)
        p = F.softmax(residual)
        p = F.dropout(p, p=0.1, training=training)
        p = residual + fc2(p)
        # print(f"p=: {p}")
        return res['loss'], p

    def predict(self, test_iterator, prob):
        out = []
        for data in test_iterator:
            out.append(self.trainer_batch.predict_batch(data, prob))
        return torch.cat(out, dim=0)

    @staticmethod
    def from_config(config):
        config['trainer_batch'] = parse_class(config['trainer_batch']['type']).from_config(config['trainer_batch'])
        return build_from_config(config)

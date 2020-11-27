"""
Neptune
-------
"""
from argparse import Namespace
from typing import Optional, List, Dict, Any, Union, Iterable


try:
    import neptune
    from neptune.experiments import Experiment
    _NEPTUNE_AVAILABLE = True
except ImportError:  # pragma: no-cover
    neptune = None
    Experiment = None
    _NEPTUNE_AVAILABLE = False

import torch
from torch import is_tensor

from pytorch_lightning import _logger as log
from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only


class NeptuneLogger(LightningLoggerBase):
    r"""
    Log using `Neptune <https://neptune.ai>`_. Install it with pip:

    .. code-block:: bash

        pip install neptune-client

    The Neptune logger can be used in the online mode or offline (silent) mode.
    To log experiment data in online mode, :class:`NeptuneLogger` requires an API key.
    In offline mode, the logger does not connect to Neptune.

    **ONLINE MODE**

    Example:
        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.loggers import NeptuneLogger
        >>> # arguments made to NeptuneLogger are passed on to the neptune.experiments.Experiment class
        >>> # We are using an api_key for the anonymous user "neptuner" but you can use your own.
        >>> neptune_logger = NeptuneLogger(
        ...     api_key='ANONYMOUS',
        ...     project_name='shared/pytorch-lightning-integration',
        ...     experiment_name='default',  # Optional,
        ...     params={'max_epochs': 10},  # Optional,
        ...     tags=['pytorch-lightning', 'mlp']  # Optional,
        ... )
        >>> trainer = Trainer(max_epochs=10, logger=neptune_logger)

    **OFFLINE MODE**

    Example:
        >>> from pytorch_lightning.loggers import NeptuneLogger
        >>> # arguments made to NeptuneLogger are passed on to the neptune.experiments.Experiment class
        >>> neptune_logger = NeptuneLogger(
        ...     offline_mode=True,
        ...     project_name='USER_NAME/PROJECT_NAME',
        ...     experiment_name='default',  # Optional,
        ...     params={'max_epochs': 10},  # Optional,
        ...     tags=['pytorch-lightning', 'mlp']  # Optional,
        ... )
        >>> trainer = Trainer(max_epochs=10, logger=neptune_logger)

    Use the logger anywhere in you :class:`~pytorch_lightning.core.lightning.LightningModule` as follows:

    >>> from pytorch_lightning import LightningModule
    >>> class LitModel(LightningModule):
    ...     def training_step(self, batch, batch_idx):
    ...         # log metrics
    ...         self.logger.experiment.log_metric('acc_train', ...)
    ...         # log images
    ...         self.logger.experiment.log_image('worse_predictions', ...)
    ...         # log model checkpoint
    ...         self.logger.experiment.log_artifact('model_checkpoint.pt', ...)
    ...         self.logger.experiment.whatever_neptune_supports(...)
    ...
    ...     def any_lightning_module_function_or_hook(self):
    ...         self.logger.experiment.log_metric('acc_train', ...)
    ...         self.logger.experiment.log_image('worse_predictions', ...)
    ...         self.logger.experiment.log_artifact('model_checkpoint.pt', ...)
    ...         self.logger.experiment.whatever_neptune_supports(...)

    If you want to log objects after the training is finished use ``close_after_fit=False``:

    .. code-block:: python

        neptune_logger = NeptuneLogger(
            ...
            close_after_fit=False,
            ...
        )
        trainer = Trainer(logger=neptune_logger)
        trainer.fit()

        # Log test metrics
        trainer.test(model)

        # Log additional metrics
        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(y_true, y_pred)
        neptune_logger.experiment.log_metric('test_accuracy', accuracy)

        # Log charts
        from scikitplot.metrics import plot_confusion_matrix
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(16, 12))
        plot_confusion_matrix(y_true, y_pred, ax=ax)
        neptune_logger.experiment.log_image('confusion_matrix', fig)

        # Save checkpoints folder
        neptune_logger.experiment.log_artifact('my/checkpoints')

        # When you are done, stop the experiment
        neptune_logger.experiment.stop()

    See Also:
        - An `Example experiment <https://ui.neptune.ai/o/shared/org/
          pytorch-lightning-integration/e/PYTOR-66/charts>`_ showing the UI of Neptune.
        - `Tutorial <https://docs.neptune.ai/integrations/pytorch_lightning.html>`_ on how to use
          Pytorch Lightning with Neptune.

    Args:
        api_key: Required in online mode.
            Neptune API token, found on https://neptune.ai.
            Read how to get your
            `API key <https://docs.neptune.ai/python-api/tutorials/get-started.html#copy-api-token>`_.
            It is recommended to keep it in the `NEPTUNE_API_TOKEN`
            environment variable and then you can leave ``api_key=None``.
        project_name: Required in online mode. Qualified name of a project in a form of
            "namespace/project_name" for example "tom/minst-classification".
            If ``None``, the value of `NEPTUNE_PROJECT` environment variable will be taken.
            You need to create the project in https://neptune.ai first.
        offline_mode: Optional default ``False``. If ``True`` no logs will be sent
            to Neptune. Usually used for debug purposes.
        close_after_fit: Optional default ``True``. If ``False`` the experiment
            will not be closed after training and additional metrics,
            images or artifacts can be logged. Also, remember to close the experiment explicitly
            by running ``neptune_logger.experiment.stop()``.
        experiment_name: Optional. Editable name of the experiment.
            Name is displayed in the experiment’s Details (Metadata section) and
            in experiments view as a column.
        upload_source_files: Optional. List of source files to be uploaded.
            Must be list of str or single str. Uploaded sources are displayed
            in the experiment’s Source code tab.
            If ``None`` is passed, the Python file from which the experiment was created will be uploaded.
            Pass an empty list (``[]``) to upload no files.
            Unix style pathname pattern expansion is supported.
            For example, you can pass ``'\*.py'``
            to upload all python source files from the current directory.
            For recursion lookup use ``'\**/\*.py'`` (for Python 3.5 and later).
            For more information see :mod:`glob` library.
        params: Optional. Parameters of the experiment.
            After experiment creation params are read-only.
            Parameters are displayed in the experiment’s Parameters section and
            each key-value pair can be viewed in the experiments view as a column.
        properties: Optional. Default is ``{}``. Properties of the experiment.
            They are editable after the experiment is created.
            Properties are displayed in the experiment’s Details section and
            each key-value pair can be viewed in the experiments view as a column.
        tags: Optional. Default is ``[]``. Must be list of str. Tags of the experiment.
            They are editable after the experiment is created (see: ``append_tag()`` and ``remove_tag()``).
            Tags are displayed in the experiment’s Details section and can be viewed
            in the experiments view as a column.
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 project_name: Optional[str] = None,
                 close_after_fit: Optional[bool] = True,
                 offline_mode: bool = False,
                 experiment_name: Optional[str] = None,
                 upload_source_files: Optional[List[str]] = None,
                 params: Optional[Dict[str, Any]] = None,
                 properties: Optional[Dict[str, Any]] = None,
                 tags: Optional[List[str]] = None,
                 **kwargs):
        if not _NEPTUNE_AVAILABLE:
            raise ImportError('You want to use `neptune` logger which is not installed yet,'
                              ' install it with `pip install neptune-client`.')
        super().__init__()
        self.api_key = api_key
        self.project_name = project_name
        self.offline_mode = offline_mode
        self.close_after_fit = close_after_fit
        self.experiment_name = experiment_name
        self.upload_source_files = upload_source_files
        self.params = params
        self.properties = properties
        self.tags = tags
        self._kwargs = kwargs
        self._experiment_id = None
        self._experiment = self._create_or_get_experiment()

        log.info(f'NeptuneLogger will work in {"offline" if self.offline_mode else "online"} mode')

    def __getstate__(self):
        state = self.__dict__.copy()

        # Experiment cannot be pickled, and additionally its ID cannot be pickled in offline mode
        state['_experiment'] = None
        if self.offline_mode:
            state['_experiment_id'] = None

        return state

    @property
    @rank_zero_experiment
    def experiment(self) -> Experiment:
        r"""
        Actual Neptune object. To use neptune features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.

        Example::

            self.logger.experiment.some_neptune_function()

        """

        # Note that even though we initialize self._experiment in __init__,
        # it may still end up being None after being pickled and un-pickled
        if self._experiment is None:
            self._experiment = self._create_or_get_experiment()

        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        for key, val in params.items():
            self.experiment.set_property(f'param__{key}', val)

    @rank_zero_only
    def log_metrics(
            self,
            metrics: Dict[str, Union[torch.Tensor, float]],
            step: Optional[int] = None
    ) -> None:
        """
        Log metrics (numeric values) in Neptune experiments.

        Args:
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Step number at which the metrics should be recorded, must be strictly increasing
        """
        assert rank_zero_only.rank == 0, 'experiment tried to log from global_rank != 0'
        for key, val in metrics.items():
            self.log_metric(key, val, step=step)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        super().finalize(status)
        if self.close_after_fit:
            self.experiment.stop()

    @property
    def save_dir(self) -> Optional[str]:
        # Neptune does not save any local files
        return None

    @property
    def name(self) -> str:
        if self.offline_mode:
            return 'offline-name'
        else:
            return self.experiment.name

    @property
    def version(self) -> str:
        if self.offline_mode:
            return 'offline-id-1234'
        else:
            return self.experiment.id

    @rank_zero_only
    def log_metric(
            self,
            metric_name: str,
            metric_value: Union[torch.Tensor, float, str],
            step: Optional[int] = None
    ) -> None:
        """
        Log metrics (numeric values) in Neptune experiments.

        Args:
            metric_name: The name of log, i.e. mse, loss, accuracy.
            metric_value: The value of the log (data-point).
            step: Step number at which the metrics should be recorded, must be strictly increasing
        """
        if is_tensor(metric_value):
            metric_value = metric_value.cpu().detach()

        if step is None:
            self.experiment.log_metric(metric_name, metric_value)
        else:
            self.experiment.log_metric(metric_name, x=step, y=metric_value)

    @rank_zero_only
    def log_text(self, log_name: str, text: str, step: Optional[int] = None) -> None:
        """
        Log text data in Neptune experiments.

        Args:
            log_name: The name of log, i.e. mse, my_text_data, timing_info.
            text: The value of the log (data-point).
            step: Step number at which the metrics should be recorded, must be strictly increasing
        """
        self.log_metric(log_name, text, step=step)

    @rank_zero_only
    def log_image(self,
                  log_name: str,
                  image: Union[str, Any],
                  step: Optional[int] = None) -> None:
        """
        Log image data in Neptune experiment

        Args:
            log_name: The name of log, i.e. bboxes, visualisations, sample_images.
            image: The value of the log (data-point).
                Can be one of the following types: PIL image, `matplotlib.figure.Figure`,
                path to image file (str)
            step: Step number at which the metrics should be recorded, must be strictly increasing
        """
        if step is None:
            self.experiment.log_image(log_name, image)
        else:
            self.experiment.log_image(log_name, x=step, y=image)

    @rank_zero_only
    def log_artifact(self, artifact: str, destination: Optional[str] = None) -> None:
        """Save an artifact (file) in Neptune experiment storage.

        Args:
            artifact: A path to the file in local filesystem.
            destination: Optional. Default is ``None``. A destination path.
                If ``None`` is passed, an artifact file name will be used.
        """
        self.experiment.log_artifact(artifact, destination)

    @rank_zero_only
    def set_property(self, key: str, value: Any) -> None:
        """
        Set key-value pair as Neptune experiment property.

        Args:
            key: Property key.
            value: New value of a property.
        """
        self.experiment.set_property(key, value)

    @rank_zero_only
    def append_tags(self, tags: Union[str, Iterable[str]]) -> None:
        """
        Appends tags to the neptune experiment.

        Args:
            tags: Tags to add to the current experiment. If str is passed, a single tag is added.
                If multiple - comma separated - str are passed, all of them are added as tags.
                If list of str is passed, all elements of the list are added as tags.
        """
        if str(tags) == tags:
            tags = [tags]  # make it as an iterable is if it is not yet
        self.experiment.append_tags(*tags)

    def _create_or_get_experiment(self):
        if self.offline_mode:
            project = neptune.Session(backend=neptune.OfflineBackend()).get_project('dry-run/project')
        else:
            session = neptune.Session.with_default_backend(api_token=self.api_key)
            project = session.get_project(self.project_name)

        if self._experiment_id is None:
            exp = project.create_experiment(
                name=self.experiment_name,
                params=self.params,
                properties=self.properties,
                tags=self.tags,
                upload_source_files=self.upload_source_files,
                **self._kwargs)
        else:
            exp = project.get_experiments(id=self._experiment_id)[0]

        self._experiment_id = exp.id
        return exp

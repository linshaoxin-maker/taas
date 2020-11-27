import inspect
from abc import ABC, abstractmethod

from pytorch_lightning.core.lightning import LightningModule


class TrainerModelHooksMixin(ABC):

    def is_function_implemented(self, f_name, model=None):
        if model is None:
            model = self.get_model()
        f_op = getattr(model, f_name, None)
        return callable(f_op)

    def is_overridden(self, method_name: str, model: LightningModule = None) -> bool:
        if model is None:
            model = self.get_model()
        super_object = LightningModule

        if not hasattr(model, method_name):
            # in case of calling deprecated method
            return False

        instance_attr = getattr(model, method_name)
        if not instance_attr:
            return False
        super_attr = getattr(super_object, method_name)

        # when code pointers are different, it was implemented
        if hasattr(instance_attr, 'patch_loader_code'):
            # cannot pickle __code__ so cannot verify if PatchDataloader
            # exists which shows dataloader methods have been overwritten.
            # so, we hack it by using the string representation
            is_overridden = instance_attr.patch_loader_code != str(super_attr.__code__)
        else:
            is_overridden = instance_attr.__code__ is not super_attr.__code__
        return is_overridden

    def has_arg(self, f_name, arg_name):
        model = self.get_model()
        f_op = getattr(model, f_name, None)
        return arg_name in inspect.signature(f_op).parameters

    @abstractmethod
    def get_model(self) -> LightningModule:
        """Warning: this is just empty shell for code implemented in other class."""

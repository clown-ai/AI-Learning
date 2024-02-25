import types
from dataclasses import dataclass, field
from typing import Tuple, Union

import torch


@dataclass
class ModuleSpec:
    """
    This is a Module Specification dataclass.

    Args:
        module (Union[Tuple, type]): A tuple describing the location of the
            module class e.g. `(module.location, ModuleClass)` or the imported
            module class itself e.g. `ModuleClass` (which is already imported
            using `from module.location import ModuleClass`).
        params (dict): A dictionary of params that need to be passed while init.
    """

    module: Union[Tuple, type]
    params: dict = field(default_factory=lambda: {})
    submodules: type = None

def build_module(spec_or_module: Union[ModuleSpec, type], *args, **kwargs):
    # If the passed `spec_or_module` is
    # a `Function`, then return it as it is
    # NOTE: to support an already initialized module add the following condition
    # `or isinstance(spec_or_module, torch.nn.Module)` to the following if check
    if isinstance(spec_or_module, types.FunctionType):
        return spec_or_module
    

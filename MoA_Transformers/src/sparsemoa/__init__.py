"""
LoRA Initialization

Portions of this file are modifications based on work created and
shared by the HuggingFace Inc. team and used according to terms
described in the Apache License 2.0.
"""
from .config import LoraConfig as SoftMoAConfig
from .layer import LoraLayer as SoftMoALayer, LinearLoraLayer as SoftMoALinearLayer
from .model import SoftMoAModel

__all__ = ["SoftMoAConfig", "SoftMoAModel"]


def __getattr__(name):
    raise AttributeError(f"Module {__name__} has no attribute {name}.")

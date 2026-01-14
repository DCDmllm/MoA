"""
LoRA Model

Portions of this file are modifications based on work created and
shared by the HuggingFace Inc. team and used according to terms
described in the Apache License 2.0.
"""
from typing import Any

from peft import PeftConfig
from torch import nn

from .config import LoraConfig

class SoftMoAModel(nn.Module):
    """
    Soft Mixture of Adapters Model
    """
    prefix: list = ["lora_", 'prompt', 'adapter']

    def __init__(self, model: nn.Module, config: LoraConfig, adapter_name: str = "default") -> None:
        """
        Initialize LoraModel

        :param model: model to be adapted
        :param config: configuration of the LoRA model
        :param adapter_name: name of the adapter
        """
        super().__init__()

        self.model = model

        self.targeted_module_names: list[str] = []
        
        if not hasattr(self, "peft_config"):
            self.peft_config = {adapter_name: config} if isinstance(config, PeftConfig) else config

        self.active_adapter = adapter_name

        self._mark_only_adapters_as_trainable(model)

        if self.peft_config[adapter_name].inference_mode:
            for n, p in model.named_parameters():
                if adapter_name in n:
                    p.requires_grad = False
  
        # Copy the peft_config in the injected model.
        self.model.peft_config = self.peft_config
    
    @property
    def active_adapters(self) -> list[str]:
        if isinstance(self.active_adapter, str):
            return [self.active_adapter]
        # is already a list of str
        return self.active_adapter
    
    def __getattr__(self, name: str) -> Any:
        """
        Forward missing attributes to the wrapped module
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
    
    def forward(self, *args: Any, **kwargs: Any):
        return self.model.forward(*args, **kwargs)
    
    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        """
        Make only adapters as trainable
        """
        for name, param in model.named_parameters():
            param.requires_grad = False

        for name, param in model.named_parameters():
            for prefix in self.prefix:
                if prefix in name:
                    param.requires_grad = True
                    # param.data = param.data.float() 

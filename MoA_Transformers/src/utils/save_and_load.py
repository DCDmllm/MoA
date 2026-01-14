"""
Saving and Loading Models

Portions of this file are modifications based on work created and
shared by the HuggingFace Inc. team and used according to terms
described in the Apache License 2.0.
"""
import warnings

from peft.utils.other import EMBEDDING_LAYER_NAMES
from peft.utils.save_and_load import has_valid_embedding_base_layer, get_embedding_layer_name

from ..utils.peft_types import PeftType


def get_peft_model_state_dict(
    model, state_dict: dict = None, adapter_name: str = "default", unwrap_compiled: bool = False,
    save_embedding_layers: str | bool = "auto"
):
    """
    Get the state dict of the PEFT model

    :param model: the PEFT model
    :param state_dict: the state dict of the model (If not provided, the state dict of the passed model will be used.)
    :param adapter_name: the name of the adapter whose state dict should be returned
    :param unwrap_compiled: whether to unwrap the model if `torch.compile` was used
    :param save_embedding_layers: if `True`, save the embedding layers in addition to adapter weights;
            if `auto`, checks the common embedding layers in config's `target_modules` when available
    :return:
    """
    if unwrap_compiled:
        model = getattr(model, "_orig_mod", model)

    config = model.peft_config[adapter_name]

    if state_dict is None:
        state_dict = model.state_dict()

    if config.peft_type in (PeftType.SOFTMOA, PeftType.SPARSEMOA):
        to_return = {k: state_dict[k] for k in state_dict if ("lora_" in k or "prompt" in k or "adapter" in k)}
    else:
        raise NotImplementedError

    if getattr(model, "modules_to_save", None) is not None:
        for key, value in state_dict.items():
            if any(f"{module_name}.modules_to_save.{adapter_name}" in key for module_name in model.modules_to_save):
                to_return[key.replace("modules_to_save.", "")] = value

    # Check the common embedding layers in `target_modules` to reset `save_embedding_layers` if necessary
    if (
        save_embedding_layers == "auto"
        and hasattr(config, "target_modules")
        and any(k in config.target_modules for k in EMBEDDING_LAYER_NAMES)
    ):
        warnings.warn("Setting `save_embedding_layers` to `True` as embedding layers found in `target_modules`.")
        save_embedding_layers = True
    elif save_embedding_layers == "auto":
        save_embedding_layers = False

    if save_embedding_layers and hasattr(model, "get_input_embeddings"):
        for layer in [model.get_input_embeddings(), model.get_output_embeddings()]:
            if config.is_prompt_learning or has_valid_embedding_base_layer(layer):
                embedding_module_name = get_embedding_layer_name(model, layer, config.is_prompt_learning)
                if embedding_module_name:
                    to_return.update({k: v for k, v in state_dict.items() if embedding_module_name in k})
    elif save_embedding_layers:
        warnings.warn("Could not identify embedding layer(s) because the model is not a model in transformers.")

    return to_return


def set_peft_model_state_dict(model, peft_model_state_dict: dict, adapter_name="default"):
    """
    Set the state dict of the PEFT model

    :param model: the PEFT model.
    :param peft_model_state_dict: the state dict of the PEFT model
    :param adapter_name: the adapter name
    :return:
    """
    config = model.peft_config[adapter_name]
    state_dict = {}

    if getattr(model, "modules_to_save", None) is not None:
        for key, value in peft_model_state_dict.items():
            if any(module_name in key for module_name in model.modules_to_save):
                for module_name in model.modules_to_save:
                    if module_name in key:
                        key = key.replace(module_name, f"{module_name}.modules_to_save.{adapter_name}")
                        break
            state_dict[key] = value
    else:
        state_dict = peft_model_state_dict

    if config.peft_type in (PeftType.SOFTMOA, PeftType.SPARSEMOA):
        print('adapter_name: ', adapter_name, '\nconfig.peft_type: ', config.peft_type)
        # peft_model_state_dict = {}
    else:
        raise NotImplementedError
    
    # print('peft_model_state_dict: ', peft_model_state_dict)
    load_result = model.load_state_dict(peft_model_state_dict, strict=False)
    # print('\nload_result', load_result)
    print('\n number of missing keys:', len(load_result.missing_keys))
    print('\n number of unexpetcted keys:', len(load_result.unexpected_keys))
    return load_result

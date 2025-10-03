from transformers.utils import PushToHubMixin
import torch
from transformers import PreTrainedModel
from peft.config import PeftConfig
from peft.tuners import LoraConfig
from typing import Any, Optional
from mlora3.mytuners import LoraModel
import os

SAFETENSORS_WEIGHTS_NAME = "adapter_model.safetensors"

class PeftModel(PushToHubMixin, torch.nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
        peft_config: PeftConfig,
        adapter_name: str = "default",
        lora_num: int = 1,
        func: str = 'softmax',
        softmax_t: float = 1.0,
        autocast_adapter_dtype: bool = True,
    ):
        super().__init__()
        self.modules_to_save = None
        self.active_adapter = adapter_name
        self.peft_type = peft_config.peft_type
        self.special_peft_forward_args = {"adapter_names"}
        self._peft_config = None
        self.base_model = LoraModel(model, {adapter_name: peft_config}, adapter_name, lora_num, func, softmax_t)
        if hasattr(self.base_model, "_cast_adapter_dtype"):
            self.base_model._cast_adapter_dtype(
                adapter_name=adapter_name, autocast_adapter_dtype=autocast_adapter_dtype
            )

    @classmethod
    def from_pretrained(
        cls,
        model: torch.nn.Module,
        lora_paths: list,
        weight_path: str = None,
        func: str = 'softmax',
        softmax_t: float = 1.0,
        adapter_name: str = "default",
        is_trainable: bool = False,
        config: Optional[PeftConfig] = None,
        autocast_adapter_dtype: bool = True,
        ephemeral_gpu_offload: bool = False,
        **kwargs: Any,
    ):
        if config is None:
            config = LoraConfig.from_pretrained(lora_paths[0], **kwargs)
        if hasattr(config, "runtime_config"):
            config.runtime_config.ephemeral_gpu_offload = ephemeral_gpu_offload
        if config.is_prompt_learning and is_trainable:
            raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")
        else:
            config.inference_mode = not is_trainable
        model = PeftModelForCausalLM(
            model, config, len(lora_paths), adapter_name, func, softmax_t, autocast_adapter_dtype=autocast_adapter_dtype
        )
        model.load_adapter(
            lora_paths, adapter_name, weight_path, is_trainable=is_trainable, autocast_adapter_dtype=autocast_adapter_dtype, **kwargs
        )
        return model

    def load_adapter(
        self,
        lora_paths: list,
        adapter_name: str,
        weight_path: str = None,
        is_trainable: bool = False,
        torch_device: Optional[str] = None,
        autocast_adapter_dtype: bool = True,
        ephemeral_gpu_offload: bool = False,
        **kwargs: Any,
    ):
        if torch_device is None:
            torch_device = ('cuda' if torch.cuda.is_available() else 'cpu')
        result = {}
        for i in range(len(lora_paths)):
            path = lora_paths[i]
            if os.path.exists(os.path.join(path, SAFETENSORS_WEIGHTS_NAME)):
                filename = os.path.join(path, SAFETENSORS_WEIGHTS_NAME)
            from safetensors import safe_open
            with safe_open(filename, framework="pt", device=torch_device) as f:
                for k in f.keys():
                    sub = k.split(".")[-1]
                    kt = k.replace(sub, f"default_{i}.{sub}")
                    result[kt] = f.get_tensor(k)
        if weight_path is not None:
            weights = torch.load(weight_path, map_location=result[kt].device, weights_only=True)
            for k in weights.keys():
                result[k] = weights[k]

        load_result = self.load_state_dict(result, strict=False)
        if hasattr(self.base_model, "_cast_adapter_dtype"):
            self.base_model._cast_adapter_dtype(
                adapter_name=adapter_name, autocast_adapter_dtype=autocast_adapter_dtype
            )
        if not is_trainable:
            self.eval()
        return load_result

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            if name == "base_model":  # see #1892: prevent infinite recursion if class is not initialized
                raise
            return getattr(self.base_model, name)


class PeftModelForCausalLM(PeftModel):
    def __init__(
        self, 
        model: torch.nn.Module, 
        peft_config: PeftConfig, 
        lora_num: int, 
        adapter_name: str = "default", 
        func: str = 'softmax', 
        softmax_t: float = 1.0,
        **kwargs
    ) -> None:
        super().__init__(model, peft_config, adapter_name, lora_num, func, softmax_t, **kwargs)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation

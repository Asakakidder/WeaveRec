from abc import ABC
import torch
from torch import nn
from typing import Union, Any, Optional
from peft.config import PeftConfig
from peft.tuners.tuners_utils import BaseTunerLayer, check_target_module_exists, onload_layer, check_adapters_to_merge
from peft.utils import _get_submodules, ModulesToSaveWrapper
from itertools import chain
import re
import math
import copy
from tqdm import tqdm

class BaseTuner(nn.Module, ABC):
    def __init__(
        self,
        model,
        peft_config: Union[PeftConfig, dict[str, PeftConfig]],
        adapter_name: str,
        lora_num: int,
        func: str = 'softmax',
        softmax_t: float = 1.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.targeted_module_names: list[str] = []
        self.peft_config = {adapter_name: peft_config} if isinstance(peft_config, PeftConfig) else peft_config
        self.active_adapter: str | list[str] = adapter_name
        self.inject_adapter(self.model, adapter_name, lora_num, func, softmax_t)
        self.model.peft_config = self.peft_config
    
    @property
    def active_adapters(self) -> list[str]:
        if isinstance(self.active_adapter, str):
            return [self.active_adapter]
        # is already a list of str
        return self.active_adapter
    
    def inject_adapter(self, model: nn.Module, adapter_name: str, lora_num: int, func: str, softmax_t: float) -> None:
        peft_config = self.peft_config[adapter_name]
        model_config = getattr(model, "config", {"model_type": "custom"})
        if hasattr(model_config, "to_dict"):
            model_config = model_config.to_dict()
        key_list = [key for key, _ in model.named_modules()]
        for key in key_list:
            if not self._check_target_module_exists(peft_config, key):
                continue
            self.targeted_module_names.append(key)
            parent, target, target_name = _get_submodules(model, key)
            self._create_and_replace(peft_config, 
                                     adapter_name, 
                                     target, 
                                     target_name, 
                                     parent, 
                                     current_key=key, 
                                     lora_num=lora_num, 
                                     func=func,
                                     softmax_t=softmax_t,
                                     )
        self.set_adapter(self.active_adapters)

class LoraModel(BaseTuner):
    prefix: str = "lora_"
    def __init__(self, model, config, adapter_name, lora_num, func, softmax_t) -> None:
        super().__init__(model, config, adapter_name, lora_num, func, softmax_t)
    
    @staticmethod
    def _check_target_module_exists(lora_config, key):
        return check_target_module_exists(lora_config, key)
    
    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        lora_num, 
        func,
        softmax_t,
    ):
        pattern_keys = list(chain(lora_config.rank_pattern.keys(), lora_config.alpha_pattern.keys()))
        target_name_key = next(filter(lambda key: re.match(rf".*\.{key}$", current_key), pattern_keys), current_key)
        r = lora_config.rank_pattern.get(target_name_key, lora_config.r)
        alpha = lora_config.alpha_pattern.get(target_name_key, lora_config.lora_alpha)
        kwargs = {
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "use_rslora": lora_config.use_rslora,
            "use_dora": lora_config.use_dora,
            "ephemeral_gpu_offload": lora_config.runtime_config.ephemeral_gpu_offload,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }
        new_module = Linear(target, adapter_name, lora_num, func, softmax_t, **kwargs)
        self._replace_module(parent, target_name, new_module, target)
    
    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        if hasattr(child, "base_layer"):
            child = child.base_layer
        if not hasattr(new_module, "base_layer"):
            if hasattr(new_module, "W_q"):
                new_module.W_q = child.W_q
            else:
                new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias
        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)
        for name, module in new_module.named_modules():
            if (self.prefix in name) or ("ranknum" in name):
                weight = (
                    child.qweight
                    if hasattr(child, "qweight")
                    else child.W_q
                    if hasattr(child, "W_q")
                    else child.weight
                    if hasattr(child, "weight")
                    else next(child.parameters())
                )
                module.to(weight.device)

    def set_adapter(self, adapter_name: str | list[str]) -> None:
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                if module.merged:
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name
    
    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            if name == "model":  # see #1892: prevent infinite recursion if class is not initialized
                raise
            return getattr(self.model, name)
    
    def merge_and_unload(
        self, progressbar: bool = False, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ) -> torch.nn.Module:
        return self._unload_and_optionally_merge(
            progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
        )
    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ):
        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            with onload_layer(target):
                if hasattr(target, "base_layer"):
                    if merge:
                        target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                    self._replace_module(parent, target_name, target.get_base_layer(), target)
                elif isinstance(target, ModulesToSaveWrapper):
                    new_module = target.modules_to_save[target.active_adapter]
                    if hasattr(new_module, "base_layer"):
                        if merge:
                            new_module.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                        new_module = new_module.get_base_layer()
                    setattr(parent, target_name, new_module)
        return self.model


class LoraLayer(BaseTunerLayer):
    adapter_layer_names = ("lora_A", "lora_B", "lora_weights")
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")
    def __init__(self, 
                 base_layer: nn.Module, 
                 lora_num: int, 
                 func: str, 
                 softmax_t: float = 1.0,
                 ephemeral_gpu_offload: bool = False,
                 **kwargs) -> None:
        self.base_layer = base_layer
        self.lora_num = lora_num
        self.func = func
        self.softmax_t = softmax_t
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        self.lora_weights = nn.ParameterDict({})
        self._disable_adapters = False
        self.merged_adapters = []
        self.use_dora: dict[str, bool] = {}
        self.lora_magnitude_vector = torch.nn.ModuleDict()  # for DoRA
        self._caches: dict[str, Any] = {}
        self.ephemeral_gpu_offload: bool = ephemeral_gpu_offload
        self.kwargs = kwargs
        base_layer = self.get_base_layer()
        in_features, out_features = base_layer.in_features, base_layer.out_features
        self.in_features = in_features
        self.out_features = out_features
    
    def update_layer(
        self, adapter_name, r, lora_alpha, lora_dropout
    ):
        # adapter_name是str，'default'
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()
        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        self.lora_A[f'{adapter_name}_0'] = nn.Linear(self.in_features, r, bias=False)
        self.lora_B[f'{adapter_name}_0'] = nn.Linear(r, self.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A[f'{adapter_name}_0'].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B[f'{adapter_name}_0'].weight)
        for i in range(1, self.lora_num):
            self.lora_A[f'{adapter_name}_{i}'] = copy.deepcopy(self.lora_A[f'{adapter_name}_0'])
            self.lora_B[f'{adapter_name}_{i}'] = copy.deepcopy(self.lora_B[f'{adapter_name}_0'])
        self.lora_weights[adapter_name] = nn.Parameter(torch.Tensor([1/self.lora_num] * self.lora_num))
        # self.lora_weights[adapter_name] = nn.Parameter(torch.rand(2))
        self.scaling[adapter_name] = lora_alpha / r
        weight = getattr(self.get_base_layer(), 'weight', None)
        device , dtype = weight.device, weight.dtype
        for adapter_layer_name in self.adapter_layer_names + self.other_param_names:
            adapter_layer = getattr(self, adapter_layer_name, None)
            if not isinstance(adapter_layer, (nn.ModuleDict, nn.ParameterDict)):
                continue
            keys = adapter_layer.keys()
            for k in keys:
                if adapter_name not in k:
                    continue
                if weight.dtype.is_floating_point or weight.dtype.is_complex:
                    adapter_layer[k] = adapter_layer[k].to(device, dtype=dtype)
                else:
                    adapter_layer[k] = adapter_layer[k].to(device)


class Linear(nn.Module, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        lora_num: int,
        func: str,
        softmax_t: float = 1.0,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer, lora_num, func, softmax_t, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        
        # 初始化LoRA, 包括融合权重等
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype
        if self.func == 'softmax':
            w = torch.softmax(self.lora_weights['default']/self.softmax_t, dim=-1)
        elif self.func == 'sigmoid':
            w = torch.sigmoid(self.lora_weights['default'])
        
        for i in range(self.lora_num):
            lora_A = self.lora_A[f'default_{i}']
            lora_B = self.lora_B[f'default_{i}']
            dropout = self.lora_dropout['default']
            scaling = self.scaling['default']
            x = x.to(lora_A.weight.dtype)
            alpha = w[i]
            result = result + alpha * lora_B(lora_A(dropout(x))) * scaling

        result = result.to(torch_result_dtype)
        return result

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return
        base_layer = self.get_base_layer()
        weight = 0
        if self.func == 'softmax':
            w = torch.softmax(self.lora_weights['default']/self.softmax_t, dim=-1)
        elif self.func == 'sigmoid':
            w = torch.sigmoid(self.lora_weights['default'])
        for name in self.lora_A.keys():
            weight_A = self.lora_A[name].weight
            weight_B = self.lora_B[name].weight
            alpha = w[weight]
            output_tensor = weight_B @ weight_A * self.scaling['default']
            base_layer.weight.data += alpha * output_tensor
            weight += 1
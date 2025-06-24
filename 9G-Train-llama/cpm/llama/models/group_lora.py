from typing import Optional, Union

from opendelta.utils.signature import get_arg_names, get_arg_names_inside_func
from opendelta.utils.name_based_addressing import *
from opendelta.basemodel import DeltaBase
import paddle.nn as nn
from opendelta import BaseDeltaConfig
import math
from dataclasses import dataclass, field
import opendelta as od
from opendelta import LoraModel
import paddle.nn.functional as F
import paddle
#定义新类继承LoraMdoel
#实现Lora的字典存储，多个lora
#重载forward函数
#考虑loraconfig, 多个lora， LoRALinear之间的交互关系
import pdb
from collections import OrderedDict
from multiprocessing.sharedctypes import Value
import os
from opendelta.delta_configs import BaseDeltaConfig
from opendelta.utils.inspect import inspect_module_statistics
from opendelta.utils.model_md5 import gen_model_hash
from opendelta.utils.signature import get_arg_names, signature
from typing import Optional, Union
from opendelta.utils.cuda import get_device
from opendelta.utils.name_based_addressing import *
import paddle.nn as nn
import paddle
from functools import wraps
# from decorator import decorate
from opendelta.utils.decorate import decorate
from opendelta.utils.structure_mapping import transform
from transformers.file_utils import PushToHubMixin
from transformers.deepspeed import deepspeed_config, is_deepspeed_zero3_enabled
from opendelta import SaveLoadMixin
from opendelta import logging
from opendelta.utils.structure_mapping import CommonStructureMap
from opendelta.utils.interactive.web import interactive
from opendelta.utils.data_parallel import new_replicate_for_data_parallel
from opendelta.utils.cuda import move_dict_to_cuda
import sys
from opendelta.delta_models.lora import LowRankLinear

import bmtrain_paddle as bmt
import paddle


@torch.jit.script
def rms_layernorm(hidden: torch.Tensor, weight: torch.Tensor, eps: float):
    old_dtype = hidden.dtype
    variance = hidden.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    hidden = (hidden * torch.rsqrt(variance + eps)).to(old_dtype)
    return hidden * weight


class LayerNorm(bmt.DistributedModule):
    """RMS LayerNorm"""

    def __init__(
        self,
        dim_norm: int,
        dtype: torch.dtype = paddle.float16,
        eps: float = 1e-5,
        init_var: float = 1.0,
    ):
        super().__init__()

        self.eps = eps
        self.dim_norm = dim_norm
        self.weight = bmt.DistributedParameter(torch.full((dim_norm,), init_var, dtype=dtype))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch_size, seq_len, dim_norm)``): Input tensor that need to be normalized.
        Return:
            :obj:`torch.Tensor` of shape ``(batch_size, seq_len, dim_norm)``: The layernorm output.
        """  # noqa: E501
        assert x.size(-1) == self.dim_norm
        return rms_layernorm(x, self.weight, self.eps)

class group_LoraConfig(BaseDeltaConfig):
    def __init__(
        self,
        lora_name="zh",
        lora_r = 8,
        lora_alpha=16,
        lora_dropout=0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        arg_names = get_arg_names_inside_func(self.__init__)
        #加一个根据args,拼接self.lora_path的功能
        for arg_name in arg_names:
            if not hasattr(self, arg_name): # the arg has not been registered in parent config
                setattr(self, arg_name, locals()[arg_name])

class gate_LoraConfig(BaseDeltaConfig):
    def __init__(
        self,
        gate_name="no_name",
        gate_lora_r = 8,
        gate_lora_alpha = 8,
        gate_lora_dropout = 0.05,
        **kwargs
    ):
        super().__init__(**kwargs)
        arg_names = get_arg_names_inside_func(self.__init__)
        #加一个根据args,拼接self.lora_path的功能
        for arg_name in arg_names:
            if not hasattr(self, arg_name): # the arg has not been registered in parent config
                setattr(self, arg_name, locals()[arg_name])


# class group_LoraLinear(nn.Module):
#     #  ------------------------------------------------------------------------------------------
#     #  Copyright (c) Microsoft Corporation. All rights reserved.
#     #  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#     #  ------------------------------------------------------------------------------------------
#     #  copy from loralib and do some refactor
#     def __init__(self,
#         in_features,
#         out_features,
#         weight,
#         loraConfig_list: List[group_LoraConfig],
#     ):
#         ##TODO:根据loraconfig_list的内容，得到参数的字典
#         super().__init__()
#         self.weight = weight #检查会不会影响更新
#         self.lora_r = {}
#         self.lora_alpha = {}
#         self.lora_dropout = nn.ModuleDict({})
#         self.lora_A = nn.ParameterDict({})
#         self.lora_B = nn.ParameterDict({})
#         self.attention_Q = nn.Parameter(weight.new_zeros((out_features, in_features)))
#         self.attention_K = nn.Parameter(weight.new_zeros((out_features, in_features)))
#         self.attention_V = nn.Parameter(weight.new_zeros((out_features, in_features)))
#         for loraConfig in loraConfig_list:
#             self.lora_r[loraConfig.lora_name] = loraConfig.lora_r
#             self.lora_alpha[loraConfig.lora_name] = loraConfig.lora_alpha
#             if loraConfig.lora_dropout > 0.:
#                 self.lora_dropout[loraConfig.lora_name] = nn.Dropout(p=loraConfig.lora_dropout)
#             else:
#                 self.lora_dropout[loraConfig.lora_name] = nn.Identity()
#             if loraConfig.lora_r > 0:
#                 self.lora_A[loraConfig.lora_name] = nn.Parameter(weight.new_zeros((loraConfig.lora_r, in_features)))
#                 self.lora_B[loraConfig.lora_name] = nn.Parameter(weight.new_zeros((out_features, loraConfig.lora_r)))
#             self.scaling[loraConfig.lora_name] = self.lora_alpha[loraConfig.lora_name] / self.lora_r[loraConfig.lora_name]
#             nn.init.kaiming_uniform_(self.lora_A[loraConfig.lora_name], a=math.sqrt(5))
#             nn.init.zeros_(self.lora_B[loraConfig.lora_name])
#     def add_LoRA():
#         pass

#     def forward(self, x):
#         batch_size, seq_len, input_dim = x.size()
#         delta_hiddens = {}
#         for lora_name in self.lora_r.keys():
#             delta_hidden = (self.lora_dropout[lora_name](x) @ self.lora_A[lora_name].T @ self.lora_B[lora_name].T) * self.scaling[lora_name]
#             delta_hiddens[lora_name] = delta_hidden.view(batch_size, seq_len, -1)
#         hidden = F.linear(x, self.weight) #llama的bias默认是False
#         # 进行attention的计算, hidden作为query, delta_hiddens作为key和value
#         attention_Q = self.attention_Q.unsqueeze(0).repeat(batch_size, 1, 1)
#         attention_K = torch.cat([self.attention_K[lora_name].unsqueeze(0).repeat(batch_size, seq_len, 1) for lora_name in self.lora_r.keys()], dim=-1)
#         attention_V = torch.cat([self.attention_V[lora_name].unsqueeze(0).repeat(batch_size, seq_len, 1) for lora_name in self.lora_r.keys()], dim=-1)
#         attention_Q = attention_Q.view(batch_size * seq_len, -1)
#         attention_K = attention_K.view(batch_size * seq_len, -1)
#         attention_V = attention_V.view(batch_size * seq_len, -1)
#         attention = torch.matmul(attention_Q, attention_K.T)
#         attention = F.softmax(attention, dim=-1)
#         attention = torch.matmul(attention, attention_V)
#         attention = attention.view(batch_size, seq_len, -1)
#         return attention
class group_LoraLinear(nn.Module):
    #防止重复load pretrained LoRA的参数
    _params_cache = {}

    @classmethod
    def load_lora_weight(cls, lora_name, file_path):
        if lora_name not in cls._params_cache:
            cls._params_cache[lora_name] = torch.load(file_path)
        return cls._params_cache[lora_name]

    def __init__(self, in_features, out_features, weight, insert_pos, loraConfig_list: List[group_LoraConfig]= []):
        super().__init__()
        # import pdb
        # pdb.set_trace()
        self.lora_weight_file_name = "pytorch_model.bin"
        #self.weight = weight #detach
        self.insert_pos = insert_pos
        self.lora_r = {}
        self.lora_alpha = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ParameterDict({})
        self.lora_B = nn.ParameterDict({})
        self.gate_lora_r = 8
        self.gate_lora_alpha = 8
        self.gate_lora_dropout = 0.05
        self.config = gate_LoraConfig()
        self.config.gate_lora_r = 8
        self.config.gate_lora_alpha = 8
        self.config.gate_lora_dropout = 0.05
        self.config.gate_name = "zh_and_code"
        
        # self.attention_Q = nn.Parameter(weight.new_zeros((out_features, out_features)))  # 更新维度
        # self.attention_K = nn.Parameter(weight.new_zeros((out_features, out_features)))  # 更新维度
        # self.attention_V = nn.Parameter(weight.new_zeros((out_features, out_features)))  # 更新维度
        ##TODO: 初始化attentionQKV & LowRank

        self.attention_gate_layernorm = LayerNorm(out_features)

        #LowRankLinear只计算自己的模块的输出，weight只用来初始化参数，dtype等等信息
        self.attention_gate_Q = LowRankLinear(in_features = out_features,
                                         out_features = out_features,
                                         weight = weight,
                                         r = self.gate_lora_r,
                                         lora_alpha = self.gate_lora_alpha,
                                         lora_dropout = self.gate_lora_dropout)
        self.attention_gate_K = LowRankLinear(in_features = out_features,
                                            out_features = out_features,
                                            weight = weight,
                                            r = self.gate_lora_r,
                                            lora_alpha = self.gate_lora_alpha,
                                            lora_dropout = self.gate_lora_dropout)
        self.attention_gate_V = LowRankLinear(in_features = out_features,
                                            out_features = out_features,
                                            weight = weight,
                                            r = self.gate_lora_r,
                                            lora_alpha = self.gate_lora_alpha,
                                            lora_dropout = self.gate_lora_dropout)
        #MultiHead才需要attention_out
        # self.attention_out = LowRankLinear(in_features = out_features,
        #                                     out_features = out_features,
        #                                     weight = weight,
        #                                     r = self.gate_lora_r,
        #                                     lora_alpha = self.gate_lora_alpha,
        #                                     lora_dropout = self.gate_lora_dropout)
        
                                    

        self.scaling = {}
        for loraConfig in loraConfig_list:
            lora_name = loraConfig.lora_name

            weight_path = loraConfig.lora_path+"/"+self.lora_weight_file_name
            Lora_weight = group_LoraLinear.load_lora_weight(lora_name, weight_path)

            
            self.lora_r[lora_name] = loraConfig.lora_r
            self.lora_alpha[lora_name] = loraConfig.lora_alpha
            if loraConfig.lora_dropout > 0.:
                self.lora_dropout[lora_name] = nn.Dropout(p=loraConfig.lora_dropout)
            else:
                self.lora_dropout[lora_name] = nn.Identity()
                
            if loraConfig.lora_r > 0:
                self.lora_A[lora_name] = nn.Parameter(Lora_weight[self.insert_pos+'.lora.lora_A'])
                self.lora_B[lora_name] = nn.Parameter(Lora_weight[self.insert_pos+'.lora.lora_B'])
                # self.lora_A[lora_name] = nn.Parameter(weight.new_zeros((loraConfig.lora_r, in_features)))  # 更新维度
                # self.lora_B[lora_name] = nn.Parameter(weight.new_zeros((out_features, loraConfig.lora_r)))  # 更新维度
            self.scaling[lora_name] = self.lora_alpha[lora_name] / self.lora_r[lora_name]

            #从checkpoint直接读取参数，无需初始化
            # nn.init.kaiming_uniform_(self.lora_A[lora_name], a=math.sqrt(5))
            # nn.init.zeros_(self.lora_B[lora_name])
        #pdb.set_trace()
        #self.weight_detached = weight.gather().detach()
        self.weight = weight

    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        delta_hiddens = {}
        for lora_name in self.lora_r.keys():
            delta_hidden = (self.lora_dropout[lora_name](x) @ self.lora_A[lora_name].T @ self.lora_B[lora_name].T) * self.scaling[lora_name]
            #对要进入attention的内容做layernorm
            delta_hidden = self.attention_gate_layernorm(delta_hidden)
            delta_hiddens[lora_name] = delta_hidden  # Removed .view() as it's not needed
        
        # Calculate hidden states, 同时不引入梯度
        # with torch.no_grad():
        #     hidden = F.linear(x, self.weight)
        with torch.no_grad():
            hidden = F.linear(x, self.weight)
        hidden = self.attention_gate_layernorm(hidden)

        #hidden = F.linear(x, self.weight)  # llama的bias默认是False

        # #######------旧版本：attention_QKV都是直接的线性层-------#######
        # # Calculate query
        # attention_Q = torch.matmul(hidden, self.attention_Q.T)
        # # Calculate attention keys and values for all delta_hiddens
        # attention_keys = torch.cat([torch.matmul(delta_hidden, self.attention_K.T) for delta_hidden in delta_hiddens.values()], dim=-1)
        # attention_values = torch.cat([torch.matmul(delta_hidden, self.attention_V.T) for delta_hidden in delta_hiddens.values()], dim=-1)
        # # Compute attention
        # attention_scores = torch.matmul(attention_Q, attention_keys.transpose(-1, -2))
        # attention_weights = F.softmax(attention_scores, dim=-1)
        # attention_output = torch.matmul(attention_weights, attention_values)

        # return attention_output

        # #######------11.17版本：attention_QKV都是LowRankLinear-------#######
        # # Calculate query
        # pdb.set_trace()
        # attention_Q = self.attention_gate_Q(hidden)  # 使用nn.Module而不是直接矩阵乘法
        # # Calculate attention keys and values for all delta_hiddens
        # attention_keys = torch.cat([self.attention_gate_K(delta_hidden) for delta_hidden in delta_hiddens.values()], dim=-1)
        # attention_values = torch.cat([self.attention_gate_V(delta_hidden) for delta_hidden in delta_hiddens.values()], dim=-1)
        # # Compute attention
        # attention_scores = torch.matmul(attention_Q, attention_keys.transpose(-1, -2))
        # attention_weights = F.softmax(attention_scores, dim=-1)
        # attention_output = torch.matmul(attention_weights, attention_values)
        # # Add attention output to the original hidden states
        # # output = hidden + attention_output
        # return attention_output

        # #######------11.20版本：延展维度-------#######
        # 假设 attention_Q 的形状是 (batch_size, 4096, 4096)
        # delta_hiddens 是一个字典，包含两个元素
        #pdb.set_trace()
        attention_Q = self.attention_gate_Q(hidden)
        attention_keys = torch.stack([self.attention_gate_K(delta_hidden) for delta_hidden in delta_hiddens.values()], dim=-1)  # (batch_size, 4096, 4096, 2)
        attention_values = torch.stack([self.attention_gate_V(delta_hidden) for delta_hidden in delta_hiddens.values()], dim=-1)  # (batch_size, 4096, 4096, 2)

        # 计算注意力分数
        # 首先，扩展 attention_Q 的维度以匹配 attention_keys
        attention_Q_expanded = attention_Q.unsqueeze(-1)  # (batch_size, 4096, 4096, 1)
        # 然后，进行矩阵乘法
        # attention_scores = torch.matmul(attention_Q_expanded, attention_keys.transpose(-2, -1))  # (batch_size, 4096, 4096, 2)
        attention_scores = torch.matmul(attention_Q_expanded.transpose(-2, -1), attention_keys)  # (batch_size, 4096, 4096, 2)

        # 计算注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, 4096, 4096, 2)

        # 计算最终的注意力输出
        #attention_output = torch.matmul(attention_weights, attention_values.transpose(-2, -1))
        attention_output = torch.matmul(attention_values,attention_weights.transpose(-2, -1))  # (batch_size, 4096, 4096, 2)
        attention_output = attention_output.sum(dim=-1)  # (batch_size, 4096, 4096)

        # 返回最终的输出
        return attention_output


        


        



            
    #     super().__init__()
    #     self.r = loraconfig.r
    #     self.lora_alpha = loraconfig.lora_alpha
    #     self.lora_dropout = loraconfig.lora_dropout
    #     if lora_dropout > 0.:
    #         self.lora_dropout = nn.Dropout(p=lora_dropout)
    #     else:
    #         self.lora_dropout = lambda x: x
    #     if r > 0:
    #         self.lora_A = nn.Parameter(weight.new_zeros((r, in_features)))
    #         self.lora_B = nn.Parameter(weight.new_zeros((out_features, r)))
    #         self.scaling = self.lora_alpha / self.r
    #         nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    #         nn.init.zeros_(self.lora_B)

    # def forward(self, x):
    #     return (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

def is_leaf_module(module):
    r"""Whether the module is a leaf module
    """
    return len([n for n,_ in module.named_children()]) == 0

        

def non_module_param(module: nn.Module):
    module_names = [n for n, _ in module.named_modules()]
    ret = []
    for n, p in module.named_parameters():
        if not is_child_key(n, module_names):
            ret.append((n,p))
    return ret

class group_LoraModel(LoraModel):
    delta_type = "lora"
    default_modified_modules = ['attn@.q@', 'attn@.v@']
    _supported_backends = ['hf', 'bmt']
    _need_pseudo_data = False
    def __init__(self, 
                 backbone_model: nn.Module,
                 Config_list: List[group_LoraConfig] = [],
                 modified_modules: Optional[List[str]] = None,
                 unfrozen_modules: Optional[List[str]] = None,
                 exclude_modules: Optional[List[str]] = None,
                 common_structure: Optional[bool] = None,
                 interactive_modify: Optional[Union[bool, int]] = False,
                 backend: Optional[str] = "hf",):
        DeltaBase.__init__(self,
                           backbone_model,
                           modified_modules=modified_modules,
                           unfrozen_modules=unfrozen_modules,
                           common_structure=common_structure,
                           interactive_modify=interactive_modify,
                           backend=backend,
                           )
        arg_names = get_arg_names_inside_func(self.__init__)
        for arg_name in arg_names:
            if not hasattr(self, arg_name): # not registered in parent class
                setattr(self, arg_name, locals()[arg_name])

        self.delta_modules = nn.ModuleList()

        self.add_all_delta_to_backbone(self.backbone_model,
                                   self.modified_modules,
                                   )
    
    def update_module(self, module: nn.Module, key: str):
        parent_ref, child_name, child_ref = self.find_module(module, key)
        #key : 'encoder.layers.0.self_att.self_attention.project_q'
        parallel_module = self.new_module_like(child_module=child_ref, key = key)
        self.insert_parallel_module(child_ref, delta_module=parallel_module, delta_name="lora")
    
    def _pseudo_data_to_instantiate(self, module):
        # no need to pass pseudo input, so overwrite it
        pass

    def new_module_like(self, child_module, key):
        in_features, out_features = child_module.in_features, child_module.out_features
        new_module = group_LoraLinear(in_features = in_features,
                                    out_features = out_features,
                                    weight = child_module.weight,
                                    insert_pos = key,
                                    loraConfig_list = self.Config_list,
                                    )
        
        
        if self.backend == "bmt":
            import bmtrain_paddle as bmt
            new_module = bmt.BMTrainModelWrapper(new_module)
        
        self.delta_modules.append(new_module)
        return new_module
    def forward():
        pass





    def _freeze_module_recursive(self,
                      module: Optional[nn.Module] = None,
                      exclude: Optional[List[str]] = None,
                      prefix=""):
        r"""[NODOC] Freeze the parameters of plm. Leave the parameters in exclude untouched.
        deltas module is filtered with ``_is_delta`` attributes because it may have parameter sharing to the main
        model, (e.g., bias term)

        Args:
            module (:obj:`nn.Module`, *optional*, default to :obj:`None`): The module of which some parts are frozen.
                If left with :obj:`None`, the function will the self.backbone_model as the module to be frozen.
            exclude (:obj:`List[str]`, *optional*, default to ``["deltas"]``): The parameters that don't need to
                be freezed. Default to all the delta parameters.
            set_state_dict (:obj:`bool`, *optional*, default to :obj:`True`): Whether setting the backbone model's state
                dict to all the parameters that still need grad.
            prefix (:obj:`str`, *optional*, default to ``""``): A parameters that are used for recursive frozen.
                Should not be changed by passing argument other than ``""``.

        """
        

        if is_leaf_module(module):
            for n, p in module.named_parameters():
                next_prefix = n if prefix == "" else ".".join([prefix,n])
                if self.find_key(next_prefix, exclude):
                    pdb.set_trace()
                    continue
                if "deltas" not in exclude or (not (hasattr(p, "_is_delta") and getattr(p, "_is_delta"))):
                    p.stop_gradient = True
            return
        else:
            # firstly freeze the non module params, then go deeper.
            params = non_module_param(module)
            for n, p in params:
                if "deltas" not in exclude or (not (hasattr(p, "_is_delta") and getattr(p, "_is_delta"))):
                    p.stop_gradient = True
            for n, c in module.named_children():
                next_prefix = n if prefix == "" else ".".join([prefix,n])
                if self.find_key(next_prefix, exclude): # if found, untouch the parameters
                    pdb.set_trace()
                    continue
                else:
                    self._freeze_module_recursive(c, exclude=exclude, prefix=next_prefix)


    #重载forward函数，使得forward可以根据delta的输出元组进行组合得到combined delta
    #了解backboneModel的forward如何和delta的forward进行组合


    ###TODO
    ###进一步修改group_LoraLinear
    ###查看backboneModel如何和delta_modules进行组合
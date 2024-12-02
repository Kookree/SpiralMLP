from dataclasses import dataclass, field
from typing import List, Any

import math
import warnings
#from collections import OrderedDict

import numpy as np
#import torch
#import torch.nn as nn
import torch.nn.functional as F
#from mmcv.cnn import Conv2d, build_activation_layer, build_norm_layer
#from mmcv.cnn.bricks.drop import build_dropout
#from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmengine.logging import MMLogger
from mmengine.model import (BaseModule, ModuleList, constant_init,
                            normal_init, trunc_normal_init)
#from mmengine.model.weight_init import trunc_normal_
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict
from torch.nn.modules.utils import _pair

from mmseg.registry import MODELS
from ..utils.embed import PatchEmbed

#########################################

#import logging
from math import pi, sin, cos, sqrt
import torch
import torch.nn as nn
import timm
from torchvision.ops.deform_conv import deform_conv2d
#from torch.nn.modules.utils import _pair
from timm.models.layers import DropPath, trunc_normal_
#from configuration import Classification

##################################################

"""functions start"""
#if not torch.cuda.is_available():
#    def get_root_logger(log_level=logging.INFO):
#        logger = logging.getLogger(__name__)
#        logger.setLevel(log_level)
#        ch = logging.StreamHandler()
#        ch.setLevel(log_level)
#        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#        ch.setFormatter(formatter)
#        logger.addHandler(ch)
#        return logger
#    
#    def load_checkpoint(model, checkpoint_path, map_location='cpu'):
#        state_dict = torch.load(checkpoint_path, map_location=map_location)
#        model.load_state_dict(state_dict) 

#class PartitionPoolingLayer(nn.Module):
#    """TODO: need to verify the function"""
#    def __init__(self, num_paritions=1, size_paritions=None):
#        super().__init__()
#        self.size_paritions = size_paritions
#        self.num_paritions = num_paritions
#
#    def forward(self, x):
#        b, c, h, w = x.shape
#        x = x.permute(0, 2, 3, 1).view(b, h*w, c)   # b, hw, c
#        size_paritions = self.size_paritions
#        if size_paritions is None:
#            size_paritions = c // self.num_paritions
#        pool = nn.functional.max_pool1d(x, kernel_size=size_paritions, stride=size_paritions)
#        pool = pool.repeat_interleave(repeats=size_paritions, dim=-1)
#        pool = pool.permute(0, 2, 1).view(b, c, h, w)
#        return pool


#class PatchEmbedOverlapping(nn.Module):
#    """
#    remove norm layer
#    """
#    def __init__(
#        self, 
#        patch_size=16, 
#        stride=16, 
#        padding=0, 
#        input_dim=3, 
#        embed_dim=768, 
##        norm_layer=None, 
#        groups=1
#        ):
#        super().__init__()
#        
#        patch_size = _pair(patch_size)
#        stride = _pair(stride)
#        padding = _pair(padding)
#        self.proj = nn.Conv2d(input_dim, embed_dim, kernel_size=patch_size, stride=stride, padding=padding, groups=groups)
##        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
#    """note: change in class SpiralNet: def forward_embedding"""
#    def forward(self, x):   # b,3,h,w
#        return self.proj(x).permute(0,2,3,1) # b,h,w,c

class Downsample(BaseModule):
    def __init__(
        self, 
        input_dim, 
        output_dim, 
        kernel_size:int=3,
        stride:int=2,
        padding:int=1,
#        patch_size:int=2
        ):
        super().__init__()

#        """related to transition"""
#        assert patch_size == 2, patch_size

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)

        self.proj = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):   # b,h,w,c
        x = self.proj(x.permute(0,3,1,2)).permute(0,2,3,1)   # b,h,w,c
        return x

"""end"""


class MLP(BaseModule):
    """
    add residual conneciton
    droprate: 0 => 0.1
    """
    def __init__(self, input_dim, hidden_dim, output_dim, act=nn.GELU, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = act()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(drop)

        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x): 
        if self.input_dim == self.output_dim:
            x_id = x
        x = self.act(self.fc1(x))
        x = self.drop(self.fc2(x))
        if self.input_dim == self.output_dim:
            x = x + x_id
        return x


class SpiralFC(BaseModule):
    def __init__(   self, 
                    config, 
                    stride:int=1, 
                    padding:int=0, 
                    dilation:int=1, 
                    kernel_size:int=3, 
                    groups:int=1, 
                    bias:bool=True
                ):
        super().__init__()

        if config.embed_dim % groups != 0:
            raise ValueError('input_dim and output_dim must be divisible by groups')
        if stride != 1:
            raise ValueError('stride must be 1')
        if padding != 0:
            raise ValueError('padding must be 0')

        self.input_dim = config.embed_dim
        self.output_dim = config.embed_dim
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        """group conv, weight tensor: [output_dim, input_dim / group, height, width]"""
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(self.output_dim, self.input_dim // self.groups, 1, 1))  # kernel size == 1

        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_dim))
        else:
            self.register_parameter('bias', None)
        self.register_buffer('offset', self.gen_offset())

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def gen_offset(self):
        """
        offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width,
            out_height, out_width]): offsets to be applied for each position in the
            convolution kernel.
        """
        offset = torch.empty(1, self.input_dim*2, 1, 1)
        k = 2
        R = 14
        num_dims = [self.input_dim//k for _ in range(k)]

        for num_dim in num_dims:
            for i in range(num_dim):
                if i <= num_dim // 2:
                    offset[0, 2 * i + 0, 0, 0] = round(i*(R/num_dim) * cos(pi * i / 16))    # along h
                    offset[0, 2 * i + 1, 0, 0] = round(i*(R/num_dim) * sin(pi * i / 16))    # along w
                else:
                    offset[0, 2 * i + 0, 0, 0] = round((R - i*(R/num_dim)) * cos(pi * i / 16))    # along h
                    offset[0, 2 * i + 1, 0, 0] = round((R - i*(R/num_dim)) * sin(pi * i / 16))    # along w
        return offset

    def forward(self, x):
        b, h, w, c = x.size()
        x = x.permute(0, 3, 1, 2)  # b,c,h,w
        return deform_conv2d(x, self.offset.expand(b, -1, h, w), self.weight, self.bias, stride=self.stride,
                                padding=self.padding, dilation=self.dilation)   # b,c,h,w

    """additional print information"""
    def extra_repr(self) -> str:
        s = self.__class__.__name__ + '('
        s += '{input_dim}'
        s += ', {output_dim}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        s += ', padding={padding}' if self.padding != (0, 0) else ''
        s += ', dilation={dilation}' if self.dilation != (1, 1) else ''
        s += ', groups={groups}' if self.groups != 1 else ''
        s += ', bias=False' if self.bias is None else ''
        s += ')'
        return s.format(**self.__dict__)


class SpiralMLP(BaseModule):
    """
    remove qkv_bias
    attn_drop: 0. => 0.1
    proj_drop: 0. => 0.1
    """
    def __init__(self, config):
        super().__init__()

        """change bias from False to True, starts"""
        self.fc_self = nn.Linear(config.embed_dim, config.embed_dim)    # why False? 
        """ends"""
#        self.fc_cross = SpiralFC(input_dim=dim, output_dim=dim, kernel_size=(3, 3))
        self.fc_cross = SpiralFC(config)

        self.reweight = MLP(config.embed_dim, config.embed_dim // 4, config.embed_dim * 2)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.proj_drop = nn.Dropout(config.proj_drop)
        """use attn_drop?"""
#        self.attn_drop = nn.Dropout(config.attn_drop)
        """use partition pooling?"""
#        self.pool = PartitionPoolingLayer(size_paritions=16)


    def forward(self, x):
        b, h, w, c = x.shape
        """use partition pooling?"""
#        x = self.pool(x.permute(0, 3, 1, 2))   # b,h,w,c

        x_cross = self.fc_cross(x).permute(0, 2, 3, 1)  # b,h,w,c
        x_self = self.fc_self(x)    # b,h,w,c

        weight = (x_cross + x_self).permute(0, 3, 1, 2).flatten(2).mean(2)
        weight = self.reweight(weight).reshape(b, c, 2).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)
#        weight = self.attn_drop(weight) # same as attn_drop in GPT after the softmax()

        x = x_cross * weight[0] + x_self * weight[1]
        return self.proj_drop(self.proj(x))

class SpiralBlock(BaseModule):
    """
    remove: 
        qkv_bias
        qk_scale
    """
    def __init__(self, config):
        super().__init__()

        self.attn = SpiralMLP(config)
        self.norm_1 = nn.LayerNorm(config.embed_dim)
        self.norm_2 = nn.LayerNorm(config.embed_dim)
        self.drop_path = DropPath(config.drop_path) if config.drop_path > 0. else nn.Identity()

        self.mlp = MLP(input_dim=config.embed_dim, hidden_dim=int(config.embed_dim*config.mlp_ratio), output_dim=config.embed_dim, act=nn.GELU)
        self.skip_lam = config.skip_lam

    def forward(self, x):   # b,h,w,c
#        x = x[0]
        x = x + self.drop_path(self.attn(self.norm_1(x))) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm_2(x))) / self.skip_lam
        return x    # b,h,w,c


@dataclass
class SpiralConfig:
    layers: List[int] 
    embed_dims: List[int] 
    swin_info: str
    layer: int = None
    embed_dim: int = None
    patch_size: int = 7
    transitions: List[bool] = field(default_factory=lambda: [True, True, True, True])
    mlp_ratio: int = None
    mlp_ratios: List[int] = field(default_factory=lambda: [4, 4, 4, 4])
    num_classes: int = 1000
    attn_drop: float = 0.1
    proj_drop: float = 0.1
    drop_path: float = 0.1
    skip_lam: float = 1.0
    fork_feat: bool = False
    pretrained: str = None
#    extra_config: Any = field(default_factory=dict)

class Classification:
    def __init__(self):
        pass
    @staticmethod
    def SpiralMLP_B1_Config():
        return SpiralConfig(layers=[2, 2, 4, 2],embed_dims=[64, 128, 320, 512], pretrained=None, swin_info="Spiral_S")

    @staticmethod
    def SpiralMLP_B2_Config():
        return SpiralConfig(layers=[2, 3, 10, 3],embed_dims=[64, 128, 320, 512], pretrained=None, swin_info="Spiral_S")

    @staticmethod
    def SpiralMLP_B3_Config():
        return SpiralConfig(layers=[3, 4, 18, 3],embed_dims=[64, 128, 320, 512], pretrained=None, swin_info="Spiral_M")

    @staticmethod
    def SpiralMLP_B4_Config():
        return SpiralConfig(layers=[3, 8, 27, 3],embed_dims=[64, 128, 320, 512], pretrained=None, swin_info="Spiral_L")

    @staticmethod
    def SpiralMLP_B5_Config():
        return SpiralConfig(layers=[3, 4, 24, 3],embed_dims=[96, 192, 384, 768], pretrained=None, swin_info="Spiral_L")



config = Classification.SpiralMLP_B5_Config()

@MODELS.register_module()
class SpiralNet(BaseModule):
    def __init__(self,                  
                 depths, 
                 pretrain_img_size=224,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=4,
#                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 5, 8],
                 patch_sizes=[4, 2, 2, 2],
                 strides=[4, 2, 2, 2],
                 paddings=[0, 0, 0, 0],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 use_abs_pos_embed=True,
                 norm_after_stage=False,
                 use_conv_ffn=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 pretrained=None,
                 convert_weights=True,
                 init_cfg=None,
                 config=config):    # init_cfg=None => start from the checkpoint
        super().__init__(init_cfg=init_cfg)
        
        """fork_feat, False=>image classification, True=>other tasks"""
        self.fork_feat = config.fork_feat
        if not self.fork_feat:
            self.num_classes = config.num_classes
#        self.patch_embed = PatchEmbedOverlapping(patch_size=config.patch_size, stride=4, padding=2, input_dim=3, embed_dim=config.embed_dims[0])
        self.patch_embed = PatchEmbed(in_channels=3, embed_dims=config.embed_dims[0], kernel_size=config.patch_size, stride=4, padding=2, bias=True, norm_cfg=dict(type='LN', eps=1e-6))
        self.embed_dims = config.embed_dims
        self.pretrained = config.pretrained
        self.stages = nn.ModuleList()

        for i, num_layers in enumerate(config.layers):
            stage = nn.ModuleList()
            for _ in range(num_layers):
                config.layer = config.layers[i]
                config.embed_dim = config.embed_dims[i]
                config.mlp_ratio = config.mlp_ratios[i]
                stage.append(SpiralBlock(config))
            if i < len(config.layers)-1 and config.transitions[i]:
                stage.append(Downsample(
                                    input_dim=config.embed_dim,
                                    output_dim=config.embed_dims[i+1],
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
#                                    patch_size
                                    ))
            self.stages.append(stage)
        self.out_indices = out_indices
        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    layer = nn.Identity()
                else:
                    layer = nn.LayerNorm(config.embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.norm = nn.LayerNorm(config.embed_dims[-1])
#            print(f"self.norm: {self.norm}")
            self.classifier_head = nn.Linear(config.embed_dims[-1], config.num_classes) if config.num_classes > 0 else nn.Identity()
        self.apply(self.cls_init_weights)
 
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, SpiralFC):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def init_weights(self):
        logger = MMLogger.get_current_instance()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(m, 0, math.sqrt(2.0 / fan_out))
                elif isinstance(m, AbsolutePositionEmbedding):
                    m.init_weights()
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            checkpoint = CheckpointLoader.load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            logger.warn(f'Load pre-trained model for '
                        f'{self.__class__.__name__} from original repo')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
#            if self.convert_weights:
#                # Because pvt backbones are not supported by mmpretrain,
#                # so we need to convert pre-trained weights to match this
#                # implementation.
#                state_dict = pvt_convert(state_dict)
#            load_state_dict(self, state_dict, strict=False, logger=logger)

#    """note: mmseg has to be with CUDA"""
#    def init_weights(self):
#        logger = get_root_logger
#        if isinstance(self.pretrained, str):
#            load_checkpoint(self, self.pretrained, map_location='cpu', strict=False, logger=logger)

    
    def get_classifier(self):
        return self.classifier_head

    def reset_classifier(self, global_pool=''):
        self.classifier_head = nn.Linear(self.embed_dims[-1], self.num_classes) if self.num_classes > 0 else nn.Identity()

    def embedding(self, img):  # b,3,h,w
        return self.patch_embed(img)   # b,h,w,c

    def processing(self, x):    # b,h,w,c
#        b,h,w,c = x.shape
        outs = []
        for idx, stage in enumerate(self.stages):
            if idx < len(self.stages)-1: 
                for block in stage[:-1]:
                    x = block(x)
                """object detection"""
                if idx in self.out_indices:
                    outs.append(x.permute(0, 3, 1, 2).contiguous())
                x = stage[-1](x)
            else: 
                for block in stage:
                    x = block(x)
                if idx in self.out_indices:
                    outs.append(x.permute(0, 3, 1, 2).contiguous())
#            for num_block, block in enumerate(stage):
#                if num_block < len(stage):
#                    x = block(x)
#                x = block(x)
        """object detection"""
        if True:
            return outs
        """image classification"""
        return x.reshape(b, -1, c)  # b,hw,c

    def forward(self, img):   # b,3,h,w
        b, c_img, h, w = img.shape
        x = self.embedding(img) # b,h,w,c
        x = self.processing(x)  # b,hw,c, in object detection: it is [(b,c,h,w)]

        """object detection"""
        if True:
#            for i, item in enumerate(x):
#                print(i)
#                print(item.shape)
#                print(f"*"*30)
#            print(f"output shape : {x[0].shape}")
            return x
#        """image classification"""
#        x = self.norm(x)
#        return self.classifier_head(x.mean(1))  # x.mean(1)=>b,c, classifier_head=>b,num_classes


#MODEL_REGISTRY = {}
#def register_model(name):
#    def decorator(cls):
#        MODEL_REGISTRY[name] = cls
#        return cls
#    return decorator
#
#@register_model('SpiralMLP_B1')
#def SpiralMLP_B1(**kwargs):
#    config = Classification.SpiralMLP_B1_Config()
#    return SpiralNet(config)
#
#@register_model('SpiralMLP_B2')
#def SpiralMLP_B2(**kwargs):
#    config = Classification.SpiralMLP_B2_Config()
#    model = SpiralNet(config)
#    return model
#
#@register_model('SpiralMLP_B3')
#def SpiralMLP_B3(**kwargs):
#    config = Classification.SpiralMLP_B3_Config()
#    model = SpiralNet(config)
#    return model
#
#@register_model('SpiralMLP_B4')
#def SpiralMLP_B4(**kwargs):
#    config = Classification.SpiralMLP_B4_Config()
#    model = SpiralNet(config)
#    return model
#
#@register_model('SpiralMLP_B5')
#def SpiralMLP_B5(**kwargs):
#    config = Classification.SpiralMLP_B5_Config()
#    model = SpiralNet(config)
#    return model

#print(f"Registed model variants: {MODEL_REGISTRY.keys()}")





if __name__ == "__main__":
    model = MODEL_REGISTRY['SpiralMLP_B1']()
    #print(model)
    x = torch.randn(2,3,224,224)
    #print(x)
    output = model.forward(x)
    print(output.shape)



#!/usr/bin/env python3


import warnings 
warnings.filterwarnings("ignore")

import torch
import math
import random
import numpy as np
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F
from einops import rearrange, repeat

from timm.models.registry import register_model
from timm.models._builder import resolve_pretrained_cfg
from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d
import torch.nn.init as init


try:
    from timm.models._builder import _update_default_kwargs as update_args
except:
    from timm.models._builder import _update_default_model_kwargs as update_args

from timm.models.registry import register_model
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import PatchEmbed, Mlp
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


def _cfg(url='', **kwargs):
    return {'url': url,
            'num_classes': 1000,
            'input_size': (3, 224, 224),
            'pool_size': None,
            'crop_pct': 0.875,
            'interpolation': 'bicubic',
            'fixed_input_size': True,
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225),
            **kwargs
            }


default_cfgs = {
    'mamba_vision_B': _cfg(url='https://huggingface.co/nvidia/MambaVision-B-1K/resolve/main/mambavision_base_1k.pth.tar',
                           crop_pct=1.0,
                           input_size=(3, 224, 224),
                           crop_mode='center'),                             
}


def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size: window size
        h_w: Height of window
        w_w: Width of window
    Returns:
        local window features (num_windows*B, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size*window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B,windows.shape[2], H, W)
    return x


def _load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata
    
    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    
    if len(err_msg) > 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)


def _load_checkpoint(model,
                    filename,
                    map_location='cpu',
                    strict=False,
                    logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = torch.load(filename, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    if sorted(list(state_dict.keys()))[0].startswith('encoder'):
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}

    del state_dict['head.weight']
    del state_dict['head.bias']
    del state_dict['norm.weight']
    del state_dict['norm.bias']
    del state_dict['norm.running_mean']
    del state_dict['norm.running_var']
    del state_dict['norm.num_batches_tracked']
    
    _load_state_dict(model, state_dict, strict, logger)
    return checkpoint



class DWConv(nn.Module):
    def __init__(self, dim=768, kernel=1, padding=0): 
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel, stride=1, padding=padding, bias=True, groups=dim)
        
    def forward(self, x):
        B, C, N = x.shape
        H, W = int(np.sqrt(x.shape[-1])), int(np.sqrt(x.shape[-1]))
        x = x.view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = rearrange(x, "b d l -> b l d")
        return x
    

class Downsample(nn.Module):
    """
    Down-sampling block"
    """

    def __init__(self,
                 dim,
                 keep_dim=False,
                 reduce=False,
                 ):
        """
        Args:
            dim: feature size dimension.
            norm_layer: normalization layer.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        if keep_dim:
            dim_out = dim
        elif reduce:
            dim_out = dim //2
        else:
            dim_out = 2 * dim
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False),
        )

    def forward(self, x):
        x = self.reduction(x)
        return x


class PatchEmbed(nn.Module):
    """
    Patch embedding block"
    """

    def __init__(self, in_chans=3, in_dim=64, dim=96):
        """
        Args:
            in_chans: number of input channels.
            dim: feature size dimension.
        """
        # in_dim = 1
        super().__init__()
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU()
            )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, dim,
                 drop_path=0.,
                 layer_scale=None,
                 kernel_size=3):
        super().__init__()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate= 'tanh')
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
        self.layer_scale = layer_scale
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        else:
            self.layer_scale = False
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, x_t=None):
        input = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)
        x = input + self.drop_path(x)

        if x_t is not None:
            input_t = x_t
            x_t = self.conv1(x_t)
            x_t = self.norm1(x_t)
            x_t = self.act1(x_t)
            x_t = self.conv2(x_t)
            x_t = self.norm2(x_t)
            if self.layer_scale:
                x_t = x_t * self.gamma.view(1, -1, 1, 1)
            x_t = input_t + self.drop_path(x_t)
            return x, x_t, x, x_t
        return x, None, None, None


class MambaVisionMixer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True, 
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )

        def _initialize_weights_conv(m, mean, stddev):
            m.dwconv.weight.data.normal_(mean, stddev)
            if m.dwconv.bias is not None:
                nn.init.zeros_(m.dwconv.bias)

        self.dwconv1 = DWConv(dim=self.d_inner//2, kernel=1, padding=0)
        _initialize_weights_conv(self.dwconv1, 0, 0.01)

        self.dwconv3 = DWConv(dim=self.d_inner//2, kernel=3, padding=1)
        _initialize_weights_conv(self.dwconv3, 0, 0.01)
        
        self.channel_cos = nn.CosineSimilarity(dim=-1)
        self.spatial_cos = nn.CosineSimilarity(dim=1)
        
    def forward(self, hidden_states, hidden_states_t=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")

        if hidden_states_t is not None:
            xz_t = self.in_proj(hidden_states_t)
            xz_t = rearrange(xz_t, "b l d -> b d l")
            x_t, z_t = xz_t.chunk(2, dim=1)
            A_t = -torch.exp(self.A_log.float())
            x_t = F.silu(F.conv1d(input=x_t, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
            z_t = F.silu(F.conv1d(input=z_t, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))
            x_dbl_t = self.x_proj(rearrange(x_t, "b d l -> (b l) d"))
            dt_t, B_t, C_t = torch.split(x_dbl_t, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt_t = rearrange(self.dt_proj(dt_t), "(b l) d -> b d l", l=seqlen)
            B_t = rearrange(B_t, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C_t = rearrange(C_t, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        if hidden_states_t is not None:
            x_t1, x_t2, x_t3, x_t4  = x_t.chunk(4, dim=1)
            x_1, x_2, x_3, x_4 = x.chunk(4, dim=1)

            x_mix = torch.cat([x_1, x_t2, x_3, x_t4], dim=1)
            x_t_mix = torch.cat([x_t1, x_2, x_t3, x_4], dim=1)

            cos_sim_channel = self.channel_cos(x_mix, x_t_mix)
            cos_sim_channel = cos_sim_channel.unsqueeze(-1)

            x_channel = x * cos_sim_channel
            x_t_channel = x_t * cos_sim_channel

            x_sp3 = self.dwconv3(x)

            x_t_sp3 = self.dwconv3(x_t)

            cos_sim_spat = self.spatial_cos(x_sp3, x_t_sp3)
            cos_sim_spat = cos_sim_spat.unsqueeze(1)
            x_spatial =  x * cos_sim_spat
            x_t_spatial = x_t * cos_sim_spat

            y_t_channel = selective_scan_fn(x_t_channel, 
                                dt_t, 
                                A_t, 
                                B_t, 
                                C_t, 
                                self.D.float(), 
                                z=None, 
                                delta_bias=self.dt_proj.bias.float(), 
                                delta_softplus=True, 
                                return_last_state=None)       
            out_t_channel = torch.cat([y_t_channel, z_t], dim=1)

            y_t_spatial = selective_scan_fn(x_t_spatial, 
                                dt_t, 
                                A_t, 
                                B_t, 
                                C_t, 
                                self.D.float(), 
                                z=None, 
                                delta_bias=self.dt_proj.bias.float(), 
                                delta_softplus=True, 
                                return_last_state=None)       
            out_t_spatial = torch.cat([y_t_spatial, z_t], dim=1)

            out_t = out_t_channel + out_t_spatial
            out_t = rearrange(out_t, "b d l -> b l d")           
            out_t = self.out_proj(out_t)

            y_channel = selective_scan_fn(x_channel, 
                            dt, 
                            A, 
                            B, 
                            C, 
                            self.D.float(), 
                            z=None, 
                            delta_bias=self.dt_proj.bias.float(), 
                            delta_softplus=True, 
                            return_last_state=None)
            out_channel = torch.cat([y_channel, z], dim=1)

            y_spatial = selective_scan_fn(x_spatial, 
                            dt, 
                            A, 
                            B, 
                            C, 
                            self.D.float(), 
                            z=None, 
                            delta_bias=self.dt_proj.bias.float(), 
                            delta_softplus=True, 
                            return_last_state=None)
            out_spatial = torch.cat([y_spatial, z], dim=1)

            out = out_channel + out_spatial
            out = rearrange(out, "b d l -> b l d")
            out = self.out_proj(out)
            return out, out_t, out, out_t

        else:
            y = selective_scan_fn(x, 
                                dt, 
                                A, 
                                B, 
                                C, 
                                self.D.float(), 
                                z=None, 
                                delta_bias=self.dt_proj.bias.float(), 
                                delta_softplus=True, 
                                return_last_state=None)
            out = torch.cat([y, z], dim=1)
            out = rearrange(out, "b d l -> b l d")
            out = self.out_proj(out)
            return out, None, None, None    
    

class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = False

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, x_t=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
             q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if x_t is not None:
            qkv_t = self.qkv(x_t).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q_t, k_t, v_t = qkv_t.unbind(0)
            q_t, k_t = self.q_norm(q_t), self.k_norm(k_t)

            if self.fused_attn:
                x_t = F.scaled_dot_product_attention(
                q_t, k_t, v_t,
                    dropout_p=self.attn_drop.p,
                )
            else:
                q_t = q_t * self.scale
                attn_t = q_t @ k_t.transpose(-2, -1)
                attn_t = attn_t.softmax(dim=-1)
                attn_t = self.attn_drop(attn_t)
                x_t = attn_t @ v_t

            x_t = x_t.transpose(1, 2).reshape(B, N, C)
            x_t = self.proj(x_t)
            x_t = self.proj_drop(x_t)

            if self.fused_attn:
                x_ts = F.scaled_dot_product_attention(
                q_t, k, v,
                    dropout_p=self.attn_drop.p,
                )
            else:
                attn_ts = q_t @ k.transpose(-2, -1)
                attn_ts = attn_ts.softmax(dim=-1)
                attn_ts = self.attn_drop(attn_ts)
                x_ts = attn_ts @ v

            x_ts = x_ts.transpose(1, 2).reshape(B, N, C)
            x_ts = self.proj(x_ts)
            x_ts = self.proj_drop(x_ts)

            if self.fused_attn:
                x_st = F.scaled_dot_product_attention(
                q, k_t, v_t,
                    dropout_p=self.attn_drop.p,
                )
            else:
                attn_st = q @ k_t.transpose(-2, -1)
                attn_st = attn_st.softmax(dim=-1)
                attn_st = self.attn_drop(attn_st)
                x_st = attn_st @ v_t

            x_st = x_st.transpose(1, 2).reshape(B, N, C)
            x_st = self.proj(x_st)
            x_st = self.proj_drop(x_st)
            return x, x_t, x_ts, x_st
        return x, None, None, None


class Block(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads, 
                 counter, 
                 transformer_blocks, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=False, 
                 drop=0., 
                 attn_drop=0.,
                 drop_path=0., 
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, 
                 Mlp_block=Mlp,
                 layer_scale=None,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if counter in transformer_blocks:
            self.mixer = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
        )
        else:
            self.mixer = MambaVisionMixer(d_model=dim, 
                                          d_state=8,  
                                          d_conv=3,    
                                          expand=1
                                          )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1

    def forward(self, x, x_t=None, x_st=None, x_ts=None):
        if x_t is not None:
            if x_st is not None and x_ts is not None:
                x_mix, xt_mix, _, _ = self.mixer(self.norm1(x), self.norm1(x_t))
                xts_mix, xst_mix, _, _ = self.mixer(self.norm1(x_ts), self.norm1(x_st))
                x, x_t, xts, xst = x + self.drop_path(self.gamma_1 * x_mix), \
                            x_t + self.drop_path(self.gamma_1 * xt_mix), \
                            x_ts + self.drop_path(self.gamma_1 * xts_mix), \
                            x_st + self.drop_path(self.gamma_1 * xst_mix)
            else:
                x_mix, xt_mix, xts_mix, xst_mix = self.mixer(self.norm1(x), self.norm1(x_t))
                x, x_t, xts, xst = x + self.drop_path(self.gamma_1 * x_mix), \
                                x_t + self.drop_path(self.gamma_1 * xt_mix), \
                                x + self.drop_path(self.gamma_1 * xts_mix), \
                                x_t + self.drop_path(self.gamma_1 * xst_mix)
            
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            x_t = x_t + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_t)))
            xts = xts + self.drop_path(self.gamma_2 * self.mlp(self.norm2(xts)))
            xst = xst + self.drop_path(self.gamma_2 * self.mlp(self.norm2(xst)))
            return x, x_t, xts, xst
        
        x_mix, _, _, _ = self.mixer(self.norm1(x))
        x = x + self.drop_path(self.gamma_1 * x_mix)
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x, None, None, None


class MambaVisionLayer(nn.Module):
    """
    MambaVision layer"
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size,
                 conv=False,
                 downsample=True,
                 reduce=False,
                 keep_dim=False,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 transformer_blocks = [],
    ):
        """
        Args:
            dim: feature size dimension.
            depth: number of layers in each stage.
            window_size: window size in each stage.
            conv: bool argument for conv stage flag.
            downsample: bool argument for down-sampling.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
            transformer_blocks: list of transformer blocks.
        """

        super().__init__()
        self.conv = conv
        self.transformer_block = False
        if conv:
            self.blocks = nn.ModuleList([ConvBlock(dim=dim,
                                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                                   layer_scale=layer_scale_conv)
                                                   for i in range(depth)])
            self.transformer_block = False
        else:
            self.blocks = nn.ModuleList([Block(dim=dim,
                                               counter=i, 
                                               transformer_blocks=transformer_blocks,
                                               num_heads=num_heads,
                                               mlp_ratio=mlp_ratio,
                                               qkv_bias=qkv_bias,
                                               qk_scale=qk_scale,
                                               drop=drop,
                                               attn_drop=attn_drop,
                                               drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                               layer_scale=layer_scale)
                                               for i in range(depth)])
            self.transformer_block = True

        self.downsample = None if not downsample else Downsample(dim=dim, reduce=reduce, keep_dim=keep_dim)
        self.do_gt = False
        self.window_size = window_size

    def forward(self, x, x_t=None, x_ts=None, x_st=None):
        _, _, H, W = x.shape
        if self.transformer_block:
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0:
                if x_t is not None:
                    x_t = torch.nn.functional.pad(x_t, (0,pad_r,0,pad_b))
                    
                x = torch.nn.functional.pad(x, (0,pad_r,0,pad_b))
                _, _, Hp, Wp = x.shape
            else:
                Hp, Wp = H, W
            if x_t is not None:
                x_t = window_partition(x_t, self.window_size)

            x = window_partition(x, self.window_size)

            if x_st is not None and x_ts is not None:
                pad_r = (self.window_size - W % self.window_size) % self.window_size
                pad_b = (self.window_size - H % self.window_size) % self.window_size
                if pad_r > 0 or pad_b > 0:
                    x_ts = torch.nn.functional.pad(x_ts, (0,pad_r,0,pad_b))
                    x_st = torch.nn.functional.pad(x_st, (0,pad_r,0,pad_b))
                    _, _, Hp, Wp = x_ts.shape
                else:
                    Hp, Wp = H, W

                x_ts = window_partition(x_ts, self.window_size)
                x_st = window_partition(x_st, self.window_size)

        for _, blk in enumerate(self.blocks):
            if x_t is not None:
                if x_ts is not None and x_st is not None and self.transformer_block:
                    x, x_t, x_ts, x_st = blk(x, x_t, x_st, x_ts)
                else:
                    x, x_t, x_ts, x_st = blk(x, x_t)
            else:
                x, _, _, _ = blk(x)

        if self.transformer_block:
            x = window_reverse(x, self.window_size, Hp, Wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()
            if x_t is not None:
                x_t = window_reverse(x_t, self.window_size, Hp, Wp)
                if pad_r > 0 or pad_b > 0:
                    x_t = x_t[:, :, :H, :W].contiguous()

                x_ts = window_reverse(x_ts, self.window_size, Hp, Wp)
                if pad_r > 0 or pad_b > 0:
                    x_ts = x_ts[:, :, :H, :W].contiguous()

                x_st = window_reverse(x_st, self.window_size, Hp, Wp)
                if pad_r > 0 or pad_b > 0:
                    x_st = x_st[:, :, :H, :W].contiguous()

        if self.downsample is None:
            if x_t is not None:
                return x, x_t, x_ts, x_st
            else:
                return x, None, None, None
        if x_t is not None:
            return self.downsample(x), self.downsample(x_t), self.downsample(x_ts), self.downsample(x_st)
        return self.downsample(x), None, None, None


class MambaVision(nn.Module):
    """
    MambaVision,
    """

    def __init__(self,
                 dim,
                 in_dim,
                 depths,
                 window_size,
                 mlp_ratio,
                 num_heads,
                 drop_path_rate=0.2,
                 in_chans=3,
                 num_classes=1000,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 **kwargs):
        """
        Args:
            dim: feature size dimension.
            depths: number of layers in each stage.
            window_size: window size in each stage.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            drop_path_rate: drop path rate.
            in_chans: number of input channels.
            num_classes: number of classes.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
        """
        super().__init__()
        num_features = int(dim * 2 ** (len(depths) - 1))
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(len(depths)):
            conv = True if (i == 0 or i == 1) else False
            level = MambaVisionLayer(dim=int(dim * 2 ** i),
                                     depth=depths[i],
                                     num_heads=num_heads[i],
                                     window_size=window_size[i],
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     conv=conv,
                                     drop=drop_rate,
                                     attn_drop=attn_drop_rate,
                                     drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                     downsample=(i < 3),
                                     layer_scale=layer_scale,
                                     layer_scale_conv=layer_scale_conv,
                                     transformer_blocks=list(range(depths[i]//2+1, depths[i])) if depths[i]%2!=0 else list(range(depths[i]//2, depths[i])),
                                     )
            self.levels.append(level)

        self.norm = nn.ModuleList()
        self.norm.append(LayerNorm2d(256))
        self.norm.append(LayerNorm2d(512))
        self.norm.append(LayerNorm2d(1024))
        self.norm.append(LayerNorm2d(1024))

        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'rpb'}

    def forward(self, x, x_t=None, rand_val=None, kd_lays=None, entropyKD=None, perturb=None):
        layers = []
        layers_t = []
        layers_ts = []
        layers_st = []

        x = self.patch_embed(x)
        if x_t is not None:
            x_t = self.patch_embed(x_t)

        entropy_loss = 0
        for i, level in enumerate(self.levels):
            if x_t is not None:
                try:
                    x, x_t, x_ts, x_st = level(x, x_t, x_ts, x_st)
                    if i in kd_lays:
                        ent_loss, xts_ent = entropyKD(x, x_t, x_ts, x_st, i)
                        entropy_loss += ent_loss.item()

                        if i == rand_val:
                            xts_ent = xts_ent.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
                            xst_ent = xst_ent.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
                            x_t = perturb(x_t, xts_ent, xst_ent)
                except:
                    x, x_t, x_ts, x_st = level(x, x_t)
                
                x = self.norm[i](x)
                x_t = self.norm[i](x_t)
                x_ts = self.norm[i](x_ts)
                x_st = self.norm[i](x_st)
                
                layers.append(x)
                layers_t.append(x_t)
                layers_ts.append(x_ts)
                layers_st.append(x_st)
            else:
                x, _, _, _ = level(x)
                x = self.norm[i](x)
                layers.append(x)
        
        if x_t is not None:
            return layers, layers_t, layers_ts, layers_st, entropy_loss
        else:
            return layers, None, None, None, 0

    def _load_state_dict(self, 
                         pretrained, 
                         strict: bool = False):
        _load_checkpoint(self, 
                         pretrained, 
                         strict=strict)



@register_model
def mamba_vision_B(pretrained=False, **kwargs):
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_B.pth.tar")
    depths = kwargs.pop("depths", [3, 3, 10, 5])
    num_heads = kwargs.pop("num_heads", [2, 4, 8, 16])
    window_size = kwargs.pop("window_size", [8, 8, 14, 7])
    dim = kwargs.pop("dim", 128)
    in_dim = kwargs.pop("in_dim", 64)
    mlp_ratio = kwargs.pop("mlp_ratio", 4)
    resolution = kwargs.pop("resolution", 224)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.3)
    layer_scale = kwargs.pop("layer_scale", 1e-5)
    pretrained_cfg = resolve_pretrained_cfg('mamba_vision_B').to_dict()
    update_args(pretrained_cfg, kwargs, kwargs_filter=None)
    model = MambaVision(depths=depths,
                        num_heads=num_heads,
                        window_size=window_size,
                        dim=dim,
                        in_dim=in_dim,
                        mlp_ratio=mlp_ratio,
                        resolution=resolution,
                        drop_path_rate=drop_path_rate,
                        layer_scale=layer_scale,
                        layer_scale_conv=None,
                        **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    if pretrained:
        if not Path(model_path).is_file():
            url = model.default_cfg['url']
            torch.hub.download_url_to_file(url=url, dst=model_path)
        model._load_state_dict(model_path)

    mamba_out = [256,512,1024,1024]
    return model, mamba_out


@register_model
def mamba_vision_L(pretrained=False, **kwargs):
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_L.pth.tar")
    depths = kwargs.pop("depths", [3, 3, 10, 5])
    num_heads = kwargs.pop("num_heads", [4, 8, 16, 32])
    window_size = kwargs.pop("window_size", [8, 8, 14, 7])
    dim = kwargs.pop("dim", 196)
    in_dim = kwargs.pop("in_dim", 64)
    mlp_ratio = kwargs.pop("mlp_ratio", 4)
    resolution = kwargs.pop("resolution", 224)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.3)
    layer_scale = kwargs.pop("layer_scale", 1e-5)
    pretrained_cfg = resolve_pretrained_cfg('mamba_vision_L').to_dict()
    update_args(pretrained_cfg, kwargs, kwargs_filter=None)
    model = MambaVision(depths=depths,
                        num_heads=num_heads,
                        window_size=window_size,
                        dim=dim,
                        in_dim=in_dim,
                        mlp_ratio=mlp_ratio,
                        resolution=resolution,
                        drop_path_rate=drop_path_rate,
                        layer_scale=layer_scale,
                        layer_scale_conv=None,
                        **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    if pretrained:
        if not Path(model_path).is_file():
            url = model.default_cfg['url']
            torch.hub.download_url_to_file(url=url, dst=model_path)
        model._load_state_dict(model_path)

    mamba_out = [256,512,1024,1024]
    return model, mamba_out
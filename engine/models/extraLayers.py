import torch
import torch.nn as nn
try:
    from models.vmamba import MambaVisionLayer
except:
    from engine.models.vmamba import MambaVisionLayer
import numpy as np
from timm.models.layers import trunc_normal_, LayerNorm2d


class ExtraLayers(nn.Module):
    def __init__(self, dim, depths, num_heads, window_size,
                 mlp_ratio, drop_path_rate, qkv_bias,
                 qk_scale, drop_rate, attn_drop_rate,
                 layer_scale, layer_scale_conv):
        super().__init__()

        self.dim = dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.drop_path_rate = drop_path_rate
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        reduce = [True, False]
        keep_dim = [False, True]
        
        self.norm = nn.ModuleList()
        self.norm.append(LayerNorm2d(512))
        self.norm.append(LayerNorm2d(512))

        self.layer = nn.ModuleList()
        for i in range(len(self.depths)):
            self.layer.append(MambaVisionLayer(dim=int(dim[i]),
                                depth=depths[i],
                                num_heads=num_heads[i],
                                window_size=window_size[i],
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                qk_scale=qk_scale,
                                conv=False,
                                reduce=reduce[i],
                                keep_dim=keep_dim[i],
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                downsample=True,
                                layer_scale=layer_scale,
                                layer_scale_conv=layer_scale_conv,
                                transformer_blocks=list(range(depths[i]//2+1, depths[i])) if depths[i]%2!=0 else list(range(depths[i]//2, depths[i])),
                                ))

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
            
    def forward(self, x, x_t=None, x_ts=None, x_st=None, rand_val=None, entropyKD=None, perturb=None):
        sources = []
        sources_t = []
        sources_ts = []
        sources_st = []

        if x_t is not None:
            x1_mamb, x1_mamb_t, x1_mamb_ts, x1_mamb_st = self.layer[0](x, x_t, x_ts, x_st)
            x1_mamb = self.norm[0](x1_mamb)
            x1_mamb_t = self.norm[0](x1_mamb_t)
            x1_mamb_ts = self.norm[0](x1_mamb_ts)
            x1_mamb_st = self.norm[0](x1_mamb_st)
            
            sources.append(x1_mamb)
            sources_t.append(x1_mamb_t)
            sources_ts.append(x1_mamb_ts)
            sources_st.append(x1_mamb_st)

            x2_mamb, x2_mamb_t, x2_mamb_ts, x2_mamb_st = self.layer[1](x1_mamb, x1_mamb_t, x1_mamb_ts, x1_mamb_st)
            x2_mamb = self.norm[1](x2_mamb)
            x2_mamb_t = self.norm[1](x2_mamb_t)
            x2_mamb_ts = self.norm[1](x2_mamb_ts)
            x2_mamb_st = self.norm[1](x2_mamb_st)
            
            ent_loss, xts_ent = entropyKD(x1_mamb, x1_mamb_t, x1_mamb_ts, x1_mamb_st, 4)
            if rand_val == 5:
                xts_ent = xts_ent.view(x2_mamb.shape[0], x2_mamb.shape[1], x2_mamb.shape[2], x2_mamb.shape[3])
                xst_ent = xst_ent.view(x2_mamb.shape[0], x2_mamb.shape[1], x2_mamb.shape[2], x2_mamb.shape[3])
                x2_mamb_t = perturb(x2_mamb_t, xts_ent, xst_ent)

            sources.append(x2_mamb)
            sources_t.append(x2_mamb_t)
            sources_ts.append(x2_mamb_ts)
            sources_st.append(x2_mamb_st)

            return sources, sources_t, sources_ts, sources_st, ent_loss
        
        x1_mamb, _, _, _ = self.layer[0](x)
        x1_mamb = self.norm[0](x1_mamb)
        sources.append(x1_mamb)

        x2_mamb, _, _, _ = self.layer[1](x1_mamb)
        x2_mamb = self.norm[1](x2_mamb)
        sources.append(x2_mamb)
        return sources, None, None, None, None
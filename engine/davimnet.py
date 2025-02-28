import torch
import torch.nn as nn
import torch.nn.functional as F

import random
from engine.models.layers import *
from data import voc_davimnet, coco_davimnet
import os
import numpy as np
from utils.lossZoo import adv_loss
from engine.models.vmamba import mamba_vision_B, mamba_vision_L
from engine.models.extraLayers import ExtraLayers
from engine.models.advNet import LocalAdv, GlobalAdv, ReverseLayerF
from engine.models.perturb import feat_perturbation


class DAVimNet(nn.Module):
    def __init__(self, size, base, head, num_classes, mamba_out):
        super(DAVimNet, self).__init__()
        self.num_classes = num_classes
        self.cfg = (coco_davimnet, voc_davimnet)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        
        self.priors = self.priorbox.forward()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.priors = self.priors.to(device)
        
        self.size = size

        self.mamba = base

        self.extras = ExtraLayers(dim=[1024,512], depths=[8,6], num_heads=[8,4],
                                  window_size=[4,2], mlp_ratio=4.,
                                  drop_path_rate=0.3, qkv_bias=True, qk_scale=None,
                                  drop_rate=0., attn_drop_rate=0., layer_scale=1e-5,
                                  layer_scale_conv=None) 
        

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.local_adapt = LocalAdv(in_channels=1024)
        self.global_adapt = GlobalAdv(in_channels=512)

        self.softmax = nn.Softmax(dim=-1)

        self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)


    def forward(self, x, x_t=None, phase='train', CE=None, FL=None, entropyKD=None):
        """Applies network layers and ops on input image(s) x.

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        loc = list()
        conf = list()

        kd_lays = [1,2,3]
        rand_val = random.choice(kd_lays)

        if x_t is not None:
            sources, sources_t, sources_ts, sources_st, kl_int = self.mamba(x, x_t, rand_val, kd_lays, entropyKD, feat_perturbation)
        else:
            sources, _, _, _, _ = self.mamba(x)       

        
        if x_t is not None:
            extras, extras_t, extras_ts, extras_st, kl_deep = self.extras(sources[-1], sources_t[-1], sources_ts[-1], 
                                                                           sources_st[-1], rand_val, entropyKD,
                                                                           feat_perturbation)
            
            sources_t = sources_t + extras_t
            sources_ts = sources_ts + extras_ts
            sources_st = sources_st + extras_st
            sources = sources + extras
        else:
            extras, _, _, _, _ = self.extras(sources[-1])
            sources = sources + extras
        
        if x_t is not None:
            entropy_loss = kl_int

            locl_ts = self.local_adapt(ReverseLayerF.apply(sources_ts[2],1.0))
            locl_st = self.local_adapt(ReverseLayerF.apply(sources_st[2],1.0))

            globl_ts = self.global_adapt(ReverseLayerF.apply(sources_ts[5],1.0))
            globl_st = self.global_adapt(ReverseLayerF.apply(sources_st[5],1.0))
    
            loss_adv = adv_loss(
                                locl_st, locl_ts,
                                globl_st, globl_ts,
                                CE, FL)

            x = sources_ts[-1]
            for (x, l, c) in zip(sources_ts, self.loc, self.conf):
                loc.append(l(x).permute(0, 2, 3, 1).contiguous())
                conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        else:
            x = sources[-1]
            for (x, l, c) in zip(sources, self.loc, self.conf):
                loc.append(l(x).permute(0, 2, 3, 1).contiguous())
                conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        
        if phase == "test":
            output = self.detect.apply(self.num_classes, 0, 200, 0.01, 0.45,
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )

        if x_t is not None:
            return output, entropy_loss, loss_adv           
        return output, sources, None

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')



def multibox(mamba_out, cfg, extras, num_classes):
    loc_layers = []
    conf_layers = []
    mamba_source = [i for i in range(len(mamba_out))]
    for k, v in enumerate(mamba_source):
        loc_layers += [nn.Conv2d(mamba_out[v],
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(mamba_out[v],
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    
    for k, v in enumerate(extras):
        loc_layers += [nn.Conv2d(v, cfg[-(k+1)]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v, cfg[-(k+1)]
                                  * num_classes, kernel_size=3, padding=1)]
        
    return (loc_layers, conf_layers)


extras = {
    '224': [512,512],
}
mbox = {
    '224': [4, 6, 6, 6, 4, 4],  
}


def build_davimnet(size=224, num_classes=21):
    base_, mamba_out = mamba_vision_B(pretrained=True)

    head_ = multibox(mamba_out,
                    mbox[str(size)], 
                    extras[str(size)],
                    num_classes)

    return DAVimNet(size, base_, head_, num_classes, mamba_out)

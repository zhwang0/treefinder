# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from .dofa_wave_dynamic_layer import Dynamic_MLP_OFA, Dynamic_MLP_Decoder
from operator import mul
from torch.nn.modules.utils import _pair
from torch.nn import Conv2d, Dropout
import numpy as np

import torch
import torch.nn as nn
import pdb
import math
from functools import reduce
import json

from timm.models.vision_transformer import PatchEmbed, Block


import torch
from torch import nn
from functools import partial

class OFAViTForSegmentation(nn.Module):
    def __init__(self, num_classes, img_size=512, patch_size=16, drop_rate=0., 
                 embed_dim=1024, depth=24, num_heads=16, wv_planes=128, mlp_ratio=4., 
                 norm_layer=nn.LayerNorm, global_pool=False):
        super(OFAViTForSegmentation, self).__init__()

        self.wv_planes = wv_planes
        self.global_pool = global_pool
        
        self.patch_embed = Dynamic_MLP_OFA(wv_planes=128, inter_dim=128, kernel_size=16, embed_dim=embed_dim)
        self.num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        
        # ViT transformer blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        
        # Decoder: Upsample the output feature map to match the input image size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, num_classes, kernel_size=1)  # Output layer with num_classes channels
        )

        
    def forward_features(self, x, wave_list):
        # embed patches
        wavelist = torch.tensor(wave_list, device=x.device).float()
        self.waves = wavelist

        x, _ = self.patch_embed(x, self.waves)

        x = x + self.pos_embed[:, 1:, :]
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Remove CLS token and reshape the sequence output into a 2D feature map
        patch_embeddings = x[:, 1:, :]
        batch_size = x.shape[0]
        patch_size = int(self.num_patches ** 0.5)
        patch_embeddings = patch_embeddings.view(batch_size, patch_size, patch_size, -1).permute(0, 3, 1, 2)  # Shape: (batch_size, embed_dim, H', W')
        
        return patch_embeddings
    
    def forward(self, x, wave_list):
        # Get patch embeddings from the ViT model
        patch_embeddings = self.forward_features(x, wave_list)
        
        # Pass the patch embeddings through the decoder to upsample to the original image size
        segmentation_output = self.decoder(patch_embeddings)
        
        return segmentation_output

    # Initialize the segmentation model
def vit_base_patch16_segmentation(num_classes=21, **kwargs):
    model = OFAViTForSegmentation(
        num_classes=num_classes,
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


class OFAViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, drop_rate=0.,
                 embed_dim=1024, depth=24, num_heads=16, wv_planes=128, num_classes=45,
                 global_pool=True, mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()

        self.wv_planes = wv_planes
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = norm_layer
            embed_dim = embed_dim
            self.fc_norm = norm_layer(embed_dim)
        else:
            self.norm = norm_layer(embed_dim)

        self.patch_embed = Dynamic_MLP_OFA(wv_planes=128, inter_dim=128, kernel_size=16, embed_dim=embed_dim)
        self.num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])

        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    def forward_features(self, x, wave_list):
        # embed patches
        wavelist = torch.tensor(wave_list, device=x.device).float()
        self.waves = wavelist

        x, _ = self.patch_embed(x, self.waves)

        x = x + self.pos_embed[:, 1:, :]
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for block in self.blocks:
            x = block(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        return outcome

    def forward_head(self, x, pre_logits=False):
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x, wave_list):
        x = self.forward_features(x, wave_list)
        x = self.forward_head(x)
        return x


def vit_small_patch16(**kwargs):
    model = OFAViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch16(**kwargs):
    model = OFAViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = OFAViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = OFAViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

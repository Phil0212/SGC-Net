import numpy as np
import torch
import torch.nn.functional as F
import math

from torch import nn
from torch.nn import Dropout
from collections import OrderedDict
from functools import reduce
from operator import mul

import matplotlib.pyplot as plt


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)

    def forward(self, x: torch.Tensor):
        out = self.attention(self.ln_1(x))
        x = x + out[0]
        x = x + self.mlp(self.ln_2(x))
        return x, out[1]
        # return x, None


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 num_tokens:int, prompt_dim:int, total_d_layer:int, out_indices:list, get_embeddings:bool):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)
        self.num_layers = layers

        self.ln_post = LayerNorm(width)
        # self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.proj = None
        self.hoi_mlp = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(width, width*2)),
            ("gelu", QuickGELU()),
            ("fc2", nn.Linear(width*2, width))
        ]))
        self.hoi_ln = LayerNorm(width)


        ## Setting of visual prompt tuning
        self.num_tokens = num_tokens 
        self.prompt_dim = prompt_dim
        self.total_d_layer = total_d_layer
        self.get_embeddings = get_embeddings
        self.out_indices = out_indices

        if get_embeddings:
            self.dgw_ln_post = LayerNorm(width)
            self.dgw_proj = nn.Parameter(scale * torch.randn(width, prompt_dim))

            self.dgw_ln_layers = nn.ModuleList([LayerNorm(width) for _ in range(len(out_indices))])
            # self.proj_layers = nn.ParameterList([nn.Parameter(scale * torch.randn(width, output_dim)) for _ in range(len(out_indices))])
            self.dgw_proj_layers =  nn.Parameter(scale * torch.randn(len(out_indices), width, prompt_dim))


        self._init_prompt(patch_size, self.num_tokens, self.prompt_dim, self.total_d_layer)
    
    def _init_prompt(self, patch, num_tokens, prompt_dim, total_d_layer):
        patch_size = []
        patch_size.append(patch)
        patch_size.append(patch)
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

        if total_d_layer > 0:
            self.dgw_prompt_embeddings = nn.Parameter(torch.zeros(1, num_tokens, prompt_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.dgw_prompt_embeddings.data, -val, val)

            if total_d_layer > 0:  # noqa
                self.dgw_deep_prompt_embeddings = nn.Parameter(torch.zeros(total_d_layer, num_tokens, prompt_dim))
                # xavier_uniform initialization
                nn.init.uniform_(self.dgw_deep_prompt_embeddings.data, -val, val)

            self.dgw_prompt_proj = nn.Linear(prompt_dim, prompt_dim)
            nn.init.kaiming_normal_(self.dgw_prompt_proj.weight, a=0, mode='fan_out') 
            self.dgw_prompt_norm = LayerNorm(prompt_dim, eps=1e-6)
            self.dgw_prompt_dropout = Dropout(0.1)


    def forward_deep_prompt(self, embedding_output, features, attention, H, W, out_last=False):
        B = embedding_output.shape[1]

        for i in range(self.num_layers):
            if i == 0:
                hidden_states, attention_weight = self.transformer.resblocks[i](embedding_output)
            elif i <= self.dgw_deep_prompt_embeddings.shape[0]:
                deep_prompt_emb = self.dgw_prompt_dropout(self.dgw_prompt_proj(self.dgw_deep_prompt_embeddings[i-1]).expand(B, -1, -1)).permute(1, 0, 2)
                hidden_states = torch.cat((
                    hidden_states[:1, :, :],
                    deep_prompt_emb,
                    hidden_states[(1+self.num_tokens):, :, :]
                ), dim=0)

                hidden_states, attention_weight = self.transformer.resblocks[i](hidden_states)
            else:
                hidden_states = torch.cat((
                    hidden_states[:1, :, :],
                    hidden_states[-(H*W):, :, :]
                ), dim=0)
                hidden_states, attention_weight = self.transformer.resblocks[i](hidden_states)

            if i in self.out_indices: 
                before_last_feats = self.dgw_prompt_norm(hidden_states)
                features.append(before_last_feats)
                attention.append(attention_weight)
                
        encoded = self.dgw_prompt_norm(hidden_states)

        return encoded, features, attention

    def forward_reverse_deep_prompt(self, embedding_output, features, H, W, out_last=False):
        B = embedding_output.shape[1]
        deep_num_no = (12 - self.dgw_deep_prompt_embeddings.shape[0])-1

        for i in range(self.num_layers):
            if i == 0:
                hidden_states = self.transformer.resblocks[i](embedding_output) 
            elif 0<i<=deep_num_no:
                hidden_states = self.transformer.resblocks[i](hidden_states) 
            else: ## with deep prompts
                deep_prompt_emb = self.dgw_prompt_dropout(self.dgw_prompt_proj(self.dgw_deep_prompt_embeddings[i-deep_num_no-1]).expand(B, -1, -1)).permute(1, 0, 2)
                hidden_states = torch.cat((
                    hidden_states[:1, :, :],
                    deep_prompt_emb,
                    hidden_states[-(H*W):, :, :]
                ), dim=0)

                hidden_states = self.transformer.resblocks[i](hidden_states)
            
            if len(self.out_indices) > 1:
                if i in self.out_indices:
                    xp = hidden_states.permute(1, 0, 2)[:, -(H*W):, :].permute(0, 2, 1).reshape(B, -1, H, W)
                    features.append(xp.contiguous())
            
            if i == (self.num_layers-2):
                before_last_feats = self.dgw_prompt_norm(hidden_states)

        encoded = self.dgw_prompt_norm(hidden_states)
        if out_last:
            return before_last_feats
        else:
            return encoded, features

    def forward(self, x: torch.Tensor, multi_scale : bool = False, f_idxs : list = []):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)


        if self.total_d_layer > 0:
            # concat prompt
            x = torch.cat((
                x[:, :1, :],
                    self.dgw_prompt_dropout(self.dgw_prompt_proj(self.dgw_prompt_embeddings).expand(B, -1, -1)),
                    x[:, 1:, :]
                ), dim=1)
            
        x = x.permute(1, 0, 2)  # NLD -> LND

        features = []
        attention_weight = []
        if self.total_d_layer == 0: #shallow
            for i, blk in enumerate(self.transformer.resblocks):
                x = blk(x)
                if len(self.out_indices) > 1:
                    if i in self.out_indices:
                        xp = x.permute(1, 0, 2)
                        features.append(xp.contiguous())

        elif self.total_d_layer > 0: # deep
            x, features, atten = self.forward_deep_prompt(x, features,attention_weight, H, W)
        elif self.total_d_layer < 0:
            x, features, atten = self.forward_reverse_deep_prompt(x, features,attention_weight, H, W)
        else:
            AttributeError('Input correct total_d_layer')

        if self.get_embeddings:

            for idx, (feature, ln_post, proj) in enumerate(zip(features, self.dgw_ln_layers, self.dgw_proj_layers)):
                feature = ln_post(feature)   
                feature = feature @ proj
                features[idx] = feature.permute(1, 0, 2)[:, 1:, :]

            x = self.ln_post(x)
            x = x @ self.dgw_proj

            features.append(x.permute(1, 0, 2)[:, 1:, :])
            

        return features, atten

import math
from collections import OrderedDict
from functools import partial

import torch
import torch.utils.checkpoint as checkpoint
from timm.models.helpers import named_apply, adapt_input_conv
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_
from timm.models.vision_transformer import _init_vit_weights, resize_pos_embed, _load_weights
from torch import nn

from utils import load_checkpoint


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward_attn(self, attn, v, B, N, C):
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward(self, x, y):
        if y != None:
            B, N, C = x.shape
            qkv_x = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            qkv_y = self.qkv(y).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q_x, k_x, v_x = qkv_x.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
            q_y, k_y, v_y = qkv_y.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

            q_x = q_x * self.scale
            q_y = q_y * self.scale

            attn_x = (q_x @ k_x.transpose(-2, -1))
            attn_y = (q_y @ k_y.transpose(-2, -1))

            x = self.forward_attn(attn_x, v_x, B, N, C)
            y = self.forward_attn(attn_y, v_y, B, N, C)
            return x, y
        else:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, y= None):
        if y != None:
            shortcut_x = x
            shortcut_y = y

            x = self.norm1(x)
            y = self.norm1(y)

            x, y = self.attn(x, y)

            x = shortcut_x + self.drop_path(x)
            y = shortcut_y + self.drop_path(y)

            x = x + self.drop_path(self.mlp(self.norm2(x)))
            y = y + self.drop_path(self.mlp(self.norm2(y)))
            return x, y
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), None))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0, attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', use_checkpoint=False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.use_checkpoint = use_checkpoint

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.hidden = nn.Linear(self.num_features, 256)
        self.head = nn.Linear(256, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, logger=None, prefix=''):
        if logger != None:
            logger.info(f"==============> Resuming form {checkpoint_path}....................")
        if checkpoint_path.endswith("npz"):
            _load_weights(self, checkpoint_path, prefix)
        else:
            load_checkpoint(self, checkpoint_path, logger)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, y=None):
        if y != None:
            x = self.patch_embed(x)
            y = self.patch_embed(y)

            cls_token_x = (self.cls_token.clone()).expand(x.shape[0], -1, -1)
            cls_token_y = (self.cls_token.clone()).expand(y.shape[0], -1, -1)

            x = torch.cat((cls_token_x, x), dim=1)
            y = torch.cat((cls_token_y, y), dim=1)

            x = self.pos_drop(x + (self.pos_embed).clone())
            y = self.pos_drop(y + (self.pos_embed).clone())
            for block in self.blocks:
                if self.use_checkpoint:
                    x, y = checkpoint.checkpoint(block, *(x, y))
                else:
                    x, y = block(x, y)
            x = self.norm(x)
            y = self.norm(y)
            return self.pre_logits(x[:, 0]), self.pre_logits(y[:, 0])
        else:
            x = self.patch_embed(x)
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_token, x), dim=1)
            x = self.pos_drop(x + self.pos_embed)
            for block in self.blocks:
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(block, *(x, None))
                else:
                    x= block(x, None)
            x = self.norm(x)
            return self.pre_logits(x[:, 0])

    def forward_fc(self, features):
        out = self.hidden(features)
        return self.head(out)

    def forward(self, x, y=None):
        if y != None:
            features_x, features_y = self.forward_features(x, y)
            output_x = self.forward_fc(features_x)
            output_y = self.forward_fc(features_y)
            return features_x, features_y, output_x, output_y
        else:
            features_x = self.forward_features(x)
            output_x = self.forward_fc(features_x)
            return features_x, output_x


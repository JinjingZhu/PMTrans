from __future__ import print_function
from re import A
from timm.models.registry import register_model
from timm import create_model
from timm.models.swin_transformer import SwinTransformer
from timm.models.layers import Mlp, PatchEmbed, DropPath
from timm.models.vision_transformer import VisionTransformer
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import torch.distributions as dists
import torch
import torch.nn.functional as F
from functools import partial
import torch.nn as nn
from loss import *
from einops import rearrange
import math

__all__ = ['ds_swin_base_patch4_window7_224', 'ds_vit_base_patch16_224', 'ds_deit_base_patch16_224', 'ds_swin_small_patch4_window7_224']
class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)

        if self.reduction:
            return loss.mean()
        else:
            return loss
    
def cosine_distance(source_hidden_features, target_hidden_features):
    "similarity between different features"
    n_s = source_hidden_features.shape[0]
    n_t = target_hidden_features.shape[0]
    temp_matrix = torch.mm(source_hidden_features, target_hidden_features.t())
    for i in range(n_s):
        vec = source_hidden_features[i]
        temp_matrix[i] /= torch.norm(vec, p=2)
    for j in range(n_t):
        vec = target_hidden_features[j]
        temp_matrix[:, j] /= torch.norm(vec, p=2)
    return temp_matrix
def convert_to_onehot(s_label, class_num):
    s_sca_label = s_label.cpu().data.numpy()
    return np.eye(class_num)[s_sca_label]

def mixup_soft_ce(pred, targets, weight, lam):
    """ mixed categorical cross-entropy loss
    """
    loss = torch.nn.CrossEntropyLoss(reduction='none')(pred, targets)
    loss = torch.sum(lam* weight* loss) / (torch.sum(weight*lam).item())
    loss = loss*torch.sum(lam)
    return loss

def mixup_supervised_dis(preds,s_label, lam):
    """ mixup_distance_in_feature_space_for_intermediate_source
    """
    label = torch.mm(s_label,s_label.t())
    mixup_loss = -torch.sum(label * F.log_softmax(preds), dim=1)
    mixup_loss = torch.sum (torch.mul(mixup_loss, lam))
    return mixup_loss

def mixup_unsupervised_dis(preds,lam):
    """ mixup_distance_in_feature_space_for_intermediate_target
    """
    label = torch.eye(preds.shape[0]).cuda()
    mixup_loss = -torch.sum(label* F.log_softmax(preds), dim=1)
    mixup_loss = torch.sum(torch.mul(mixup_loss,lam))
    return mixup_loss

def mix_token(s_token,t_token,s_lambda):
    s_token = torch.einsum('BNC,BN -> BNC', s_token, s_lambda)
    t_token = torch.einsum('BNC,BN -> BNC', t_token, 1-s_lambda)
    m_tokens =s_token+t_token
    return m_tokens

def mix_lambda_atten(s_scores,t_scores,s_lambda,num_patch):
    t_lambda = 1-s_lambda
    if s_scores is None or t_scores is None:
        s_lambda = torch.sum(s_lambda, dim=1)/num_patch # important for /self.num_patch
        t_lambda = torch.sum(t_lambda, dim=1)/num_patch
        s_lambda = s_lambda/(s_lambda+t_lambda)        
    else:
        s_lambda = torch.sum(torch.mul(s_scores, s_lambda), dim=1)/num_patch # important for /self.num_patch
        t_lambda = torch.sum(torch.mul(t_scores, t_lambda), dim=1)/num_patch
        s_lambda = s_lambda/(s_lambda+t_lambda)
    return s_lambda

def mix_lambda (s_lambda,t_lambda):
    return torch.sum(s_lambda,dim=1) / (torch.sum(s_lambda,dim=1) + torch.sum(t_lambda,dim=1))

class Swin(SwinTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward_features(self, x, patch=False):
        if not patch:
            x = self.patch_embed(x)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x = self.layers(x)
        x = self.norm(x)  # B L C
        patch = x
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x, patch, None
        
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

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        save = attn
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, save    
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

    def forward(self, x):
        t, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(t)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn    
    
class Vit(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
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
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
        self.init_weights(weight_init)

    def forward_features(self, x, patch=False):
        if not patch:
            x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        attns = []
        for b in self.blocks:
            x, attn = b(x)
            if self.dist_token is not None:
                attns.append(attn[:,:,0,2:])
            else:
                attns.append(attn[:,:,0,1:])
        attns = torch.mean(torch.stack(attns, dim=0), dim=2) #-> ave head [12 B N-]
        attns = torch.mean(attns, dim=0)
        x = self.norm(x)       
        if self.dist_token is not None:  
            return x[:, 0], x[:, 2:], attns     
        else:
            return x[:, 0], x[:, 1:], attns      

def softplus(x):
    return  torch.log(1+torch.exp(x))


class PMTrans(nn.Module):
    arch_zoo = ['swin_base', 'vit_base', 'deit_base']
    
    def __init__(self, num_classes=65, model_name='swin_base'):
        super(PMTrans, self).__init__()
        self.model_name = model_name
        if isinstance(model_name, str):
            model_name = model_name.lower()
            assert model_name in self.arch_zoo, \
                f'Arch {model_name} is not in default archs {set(self.arch_zoo)}'
            if model_name == 'swin_base':
                model_name = 'ds_swin_base_patch4_window7_224'
            elif model_name == 'vit_base':
                model_name = 'ds_vit_base_patch16_224'
            elif model_name == 'deit_base':    
                model_name = 'ds_deit_base_patch16_224'
                
        self.backbone = create_model(model_name, pretrained=True)
        self.feature_dim = self.backbone.num_features
        self.num_patch = self.backbone.patch_embed.num_patches
        self.num_classes = num_classes

        self.my_fc_head = Mlp(self.feature_dim, hidden_features=self.feature_dim*2, out_features=self.num_classes, act_layer=nn.GELU, drop=.2)
        self.s_dist_alpha = nn.Parameter(torch.Tensor([1]))
        self.s_dist_beta = nn.Parameter(torch.Tensor([1]))
        self.super_ratio = nn.Parameter(torch.Tensor([-2]))
        self.unsuper_ratio = nn.Parameter(torch.Tensor([-2]))
 
    def attn_map(self, patch=None, label=None, attn=None, mlp=False):
        # semantic is only used in vit, deit -> no cls
        if self.model_name.startswith('swin'):
            assert(patch is not None)
            assert(label is not None)
            if mlp:
                weights = torch.einsum('DP, CD -> PC', self.my_fc_head.fc1.weight, self.my_fc_head.fc2.weight)
                scores_ = torch.einsum('BND,DC->BNC',patch, weights)
            else:
                scores_ = torch.einsum('BND,CD->BNC',patch, self.my_fc_head.weight)
            scores = torch.zeros((patch.size(0), scores_.size(1)), device=patch.device)# B, 49
        elif self.model_name.startswith('deit') or self.model_name.startswith('vit'):
            scores = attn
            
        n_p_e = int(np.sqrt(self.num_patch))
        n_p_f = int(np.sqrt(scores.size(1)))
        if attn is None:
            for b in range(patch.size(0)):
                scores[b] = scores_[b,:,label[b]]  
        scores = F.interpolate(rearrange(scores, 'B (H W) -> B 1 H W', H = n_p_f), size=(n_p_e, n_p_e)).squeeze(1)
        scores = rearrange(scores, 'B H W -> B (H W)')
        return scores.softmax(dim=-1)

    def mix_source_target(self,s_token,t_token,s_lambda,t_lambda,pred,infer_label,s_logits,t_logits,s_scores,t_scores,mem_fea,img_idx,mem_cls,weight_tgt, weight_src,):
        m_s_t_tokens = mix_token(s_token, t_token, s_lambda)        
        m_s_t_logits, m_s_t_p, _ = self.backbone.forward_features(m_s_t_tokens, patch=True)
        m_s_t_pred = self.my_fc_head(m_s_t_logits)
        t_scores = (torch.ones(32,self.num_patch)/self.num_patch).cuda()
        s_lambda = mix_lambda_atten(s_scores,t_scores,s_lambda,self.num_patch)#with attention map
       # s_lambda = mix_lambda(s_lambda, t_lambda) # without attention map
        t_lambda = 1 - s_lambda

        s_onehot = torch.tensor(convert_to_onehot(infer_label, m_s_t_pred.shape[1]), dtype=torch.float32).cuda()
        m_s_t_s = cosine_distance(m_s_t_logits, s_logits)
        m_s_t_s_similarity = mixup_supervised_dis(m_s_t_s, s_onehot, s_lambda)
        m_s_t_t = cosine_distance(m_s_t_logits, t_logits)
        m_s_t_t_similarity = mixup_unsupervised_dis(m_s_t_t, t_lambda)
        feature_space_loss= (m_s_t_s_similarity + m_s_t_t_similarity) / torch.sum(s_lambda + t_lambda)
        super_m_s_t_s_loss = mixup_soft_ce(m_s_t_pred, infer_label,weight_src, s_lambda)
        unsuper_m_s_t_loss = mixup_soft_ce(m_s_t_pred,      pred  ,weight_tgt, t_lambda)
        label_space_loss  = (super_m_s_t_s_loss + unsuper_m_s_t_loss)/torch.sum(s_lambda + t_lambda)

        return feature_space_loss,label_space_loss


    def forward(self, target, infer_label, source=None, mem_fea=None, mem_cls=None, class_weight_src=None,
                img_idx=None, source_only=False):
        # source_only: run the backbone by only the source domain, then infer it by target
        device = target.device
        B = target.shape[0]
        t_token = self.backbone.patch_embed(target)
        t_logits, t_p, t_attn = self.backbone.forward_features(t_token, patch=True)
        t_pred = self.my_fc_head(t_logits)
        t_cls = t_pred.softmax(dim=-1)

        if self.training:
            s_token = self.backbone.patch_embed(source)
            s_logits, s_p, s_attn = self.backbone.forward_features(s_token, patch=True)
            s_pred = self.my_fc_head(s_logits)
                
            if self.model_name.startswith('swin'):
                s_scores = self.attn_map(patch=s_p, label=infer_label, mlp=True)
            elif self.model_name.startswith('vit') or self.model_name.startswith('deit'):
                s_scores = self.attn_map(attn=s_attn)
                

            src_ = CrossEntropyLabelSmooth(reduction='none', num_classes=self.num_classes,
                                                     epsilon=0.1)(s_pred, infer_label)            
            if source_only:
                return src_.mean()
            
            dis = -torch.mm(t_logits.detach(), mem_fea.t())
            for di in range(dis.size(0)):
                dis[di, img_idx[di]] = torch.max(dis)
            _, p1 = torch.sort(dis, dim=1)
            w = torch.zeros(t_logits.size(0), mem_fea.size(0)).cuda()
            for wi in range(w.size(0)):
                for wj in range(5):
                    w[wi][p1[wi, wj]] = 1 / 5
            weight_tgt, pred = torch.max(w.mm(mem_cls), 1)

            weight_src = class_weight_src[infer_label].unsqueeze(0)

            classifier_loss = torch.sum(weight_src * src_) / (torch.sum(weight_src).item())

            if self.model_name.startswith('swin'):
                t_scores = self.attn_map(patch=t_p, label=pred, mlp=True)
            elif self.model_name.startswith('vit') or self.model_name.startswith('deit'):
                t_scores = self.attn_map(attn=t_attn)

            t_lambda = dists.Beta(softplus(self.s_dist_alpha), softplus(self.s_dist_beta)).rsample((B, self.num_patch,)).to(device).squeeze(-1)
            s_lambda = 1 - t_lambda

            super_m_s_t_loss, unsuper_m_s_t_loss = self.mix_source_target(s_token,t_token,s_lambda,t_lambda,pred,infer_label,s_logits,t_logits,s_scores,t_scores,mem_fea,img_idx,mem_cls,weight_tgt,weight_src,)
            total_loss= classifier_loss+softplus(self.super_ratio)*super_m_s_t_loss + softplus(self.unsuper_ratio) * unsuper_m_s_t_loss
            return total_loss
        else:
            return t_cls, t_logits
    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, logger=None):
        if logger != None:
            logger.info(f"==============> Resuming form {checkpoint_path}....................")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_dict = self.state_dict()
        pretrained_dict = checkpoint['model']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        msg = self.load_state_dict(pretrained_dict, strict=True)
        if logger != None:
            logger.info(msg)

@register_model
def ds_swin_base_patch4_window7_224(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs)
    model = Swin(**model_kwargs)
    if pretrained:
        pre = create_model('swin_base_patch4_window7_224', pretrained=pretrained)
        model.load_state_dict(pre.state_dict())
        
    return model


@register_model
def ds_swin_small_patch4_window7_224(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), **kwargs)
    model = Swin(**model_kwargs)
    if pretrained:
        pre = create_model('swin_small_patch4_window7_224', pretrained=pretrained)
        model.load_state_dict(pre.state_dict())
        
    return model

@register_model
def ds_vit_base_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = Vit(**model_kwargs)
    if pretrained:
        pre = create_model('vit_base_patch16_224', pretrained=pretrained)
        model.load_state_dict(pre.state_dict())
    return model

@register_model
def ds_deit_base_patch16_224(pretrained=False, **kwargs):
    """ DeiT-base distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, distilled=True, **kwargs)
    model = Vit(**model_kwargs)
    if pretrained:
        pre = create_model('deit_base_distilled_patch16_224', pretrained=pretrained)
        model.load_state_dict(pre.state_dict())
    return model

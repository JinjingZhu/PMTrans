# --------------------------------------------------------
# Reference from https://github.com/microsoft/Swin-Transformer
# --------------------------------------------------------
# from .swin_transformer import SwinTransformer
# from .swin_transformer_BCAT import SwinTransformerBCAT
# from .vit import VisionTransformer
# from .vit_2 import VisionTransformer as vit2
# from .vit_BCAT import VisionTransformer as VisionTransformerBCAT
from .swin_pm import PMTrans

# def build_model(config, logger):
#     model_type = config.MODEL.TYPE
#     if model_type == 'swin':
#         model_t = SwinTransformerBCAT(img_size=config.DATA.IMG_SIZE,
#                                       patch_size=config.MODEL.SWIN.PATCH_SIZE,
#                                       in_chans=config.MODEL.SWIN.IN_CHANS,
#                                       num_classes=config.MODEL.NUM_CLASSES,
#                                       embed_dim=config.MODEL.SWIN.EMBED_DIM,
#                                       depths=config.MODEL.SWIN.DEPTHS,
#                                       num_heads=config.MODEL.SWIN.NUM_HEADS,
#                                       window_size=config.MODEL.SWIN.WINDOW_SIZE,
#                                       mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
#                                       qkv_bias=config.MODEL.SWIN.QKV_BIAS,
#                                       qk_scale=config.MODEL.SWIN.QK_SCALE,
#                                       drop_rate=config.MODEL.DROP_RATE,
#                                       drop_path_rate=config.MODEL.DROP_PATH_RATE,
#                                       ape=config.MODEL.SWIN.APE,
#                                       patch_norm=config.MODEL.SWIN.PATCH_NORM,
#                                       use_checkpoint=config.TRAIN.USE_CHECKPOINT)

#         model_s = SwinTransformer(img_size=config.DATA.IMG_SIZE,
#                                 patch_size=config.MODEL.SWIN.PATCH_SIZE,
#                                 in_chans=config.MODEL.SWIN.IN_CHANS,
#                                 num_classes=config.MODEL.NUM_CLASSES,
#                                 embed_dim=config.MODEL.SWIN.EMBED_DIM,
#                                 depths=config.MODEL.SWIN.DEPTHS,
#                                 num_heads=config.MODEL.SWIN.NUM_HEADS,
#                                 window_size=config.MODEL.SWIN.WINDOW_SIZE,
#                                 mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
#                                 qkv_bias=config.MODEL.SWIN.QKV_BIAS,
#                                 qk_scale=config.MODEL.SWIN.QK_SCALE,
#                                 drop_rate=config.MODEL.DROP_RATE,
#                                 drop_path_rate=config.MODEL.DROP_PATH_RATE,
#                                 ape=config.MODEL.SWIN.APE,
#                                 patch_norm=config.MODEL.SWIN.PATCH_NORM,
#                                 use_checkpoint=config.TRAIN.USE_CHECKPOINT)

#     elif model_type == 'vit':
#         model_t = VisionTransformerBCAT(img_size=config.DATA.IMG_SIZE,
#                                         num_classes=config.MODEL.NUM_CLASSES,
#                                         embed_dim=config.MODEL.VIT.EMBED_DIM,
#                                         use_checkpoint=config.TRAIN.USE_CHECKPOINT,
#                                         patch_size=config.MODEL.VIT.PATCH_SIZE,
#                                         depth=config.MODEL.VIT.DEPTH,
#                                         num_heads=config.MODEL.VIT.NUM_HEADS,
#                                         drop_rate=config.MODEL.DROP_RATE)

#         model_s = VisionTransformer(img_size=config.DATA.IMG_SIZE,
#                                     num_classes=config.MODEL.NUM_CLASSES,
#                                     embed_dim=config.MODEL.VIT.EMBED_DIM,
#                                     use_checkpoint=config.TRAIN.USE_CHECKPOINT,
#                                     patch_size=config.MODEL.VIT.PATCH_SIZE,
#                                     depth=config.MODEL.VIT.DEPTH,
#                                     num_heads=config.MODEL.VIT.NUM_HEADS,
#                                     drop_rate=config.MODEL.DROP_RATE)

#     else:
#         raise NotImplementedError(f"Unkown model: {model_type}")

#     if config.MODEL.RESUME:
#         model_t.load_pretrained(checkpoint_path=config.MODEL.RESUME, logger=logger)
#         model_s.load_pretrained(checkpoint_path=config.MODEL.RESUME, logger=logger)

#     return model_t, model_s
def build_model(config, logger):
    model_type = config.MODEL.TYPE
    num_class = config.MODEL.NUM_CLASSES
    if model_type == 'swin':
        model = PMTrans(num_classes=num_class)
    elif model_type == 'deit':
        model = PMTrans(num_classes=num_class, model_name='deit_base')
    elif model_type == 'vit':
        model = PMTrans(num_classes=num_class, model_name='vit_base')    
        
    if config.MODEL.RESUME:
        model.load_pretrained(checkpoint_path=config.MODEL.RESUME, logger=logger)  
          
    return model
# def build_base(config, logger):
#     model = vit2(img_size=config.DATA.IMG_SIZE,
#                                 num_classes=config.MODEL.NUM_CLASSES,
#                                 embed_dim=config.MODEL.VIT.EMBED_DIM,
#                                 use_checkpoint=config.TRAIN.USE_CHECKPOINT,
#                                 patch_size=config.MODEL.VIT.PATCH_SIZE,
#                                 depth=config.MODEL.VIT.DEPTH,
#                                 num_heads=config.MODEL.VIT.NUM_HEADS,
#                                 drop_rate=config.MODEL.DROP_RATE)
#     model.load_pretrained(checkpoint_path=config.MODEL.RESUME, logger=logger)
#     return model

# --------------------------------------------------------
# Reference from https://github.com/microsoft/Swin-Transformer
# --------------------------------------------------------

from torch import optim

from utils.utils import get_parameters


def build_optimizer(config, model):
    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    parameters = get_parameters(model, lr=config.TRAIN.BASE_LR,
                                weight_decay=config.TRAIN.WEIGHT_DECAY,
                                bias_weight_decay=config.TRAIN.BIAS_WEIGHT_DECAY,
                                lr_mult=config.TRAIN.LR_MULT)
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)

    return optimizer
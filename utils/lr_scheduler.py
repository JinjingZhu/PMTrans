# --------------------------------------------------------
# Reference from https://github.com/microsoft/Swin-Transformer
# --------------------------------------------------------

from timm.scheduler.cosine_lr import CosineLRScheduler


def build_scheduler(config, optimizer):
    warmup_lr_init=config.TRAIN.BASE_LR * config.TRAIN.WARMUP_LR_MULT
    lr_min = config.TRAIN.BASE_LR * config.TRAIN.MIN_LR_MULT

    t_initial = config.TRAIN.EPOCHS
    warmup_t = config.TRAIN.WARMUP_EPOCHS

    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=config.TRAIN.EPOCHS,
        t_mul=1.,
        lr_min=lr_min,
        decay_rate=0.1,
        warmup_lr_init=warmup_lr_init,
        warmup_t=config.TRAIN.WARMUP_EPOCHS,
        cycle_limit=1,
        t_in_epochs=False,
    )

    return lr_scheduler

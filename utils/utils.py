# --------------------------------------------------------
# Reference from https://github.com/microsoft/Swin-Transformer
# --------------------------------------------------------

import os

import torch
import torch.distributed as dist


def get_parameters(model, lr, weight_decay, bias_weight_decay=0, lr_mult=2):
    params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if name.startswith("head") or name.startswith("hidden"):
            if len(param.shape) == 1 or name.endswith(".bias"):
                params += [{"params": [param], "lr": lr * lr_mult, "weight_decay": bias_weight_decay}]
            else:
                params += [{"params": [param], "lr": lr * lr_mult, "weight_decay": weight_decay}]
        else:
            if len(param.shape) == 1 or name.endswith(".bias"):
                params += [{"params": [param], "lr": lr, "weight_decay": bias_weight_decay}]
            else:
                params += [{"params": [param], "lr": lr, "weight_decay": weight_decay}]
    return params


def load_checkpoint(model, checkpoint_path, logger=None):
    if logger != None:
        logger.info(f"==============> Resuming form {checkpoint_path}....................")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_dict = model.state_dict()
    pretrained_dict = checkpoint['model']
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    msg = model.load_state_dict(pretrained_dict, strict=False)
    if logger != None:
        logger.info(msg)


def save_checkpoint(config, epoch, model, logger):
    save_state = {'model': model.state_dict()}
    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}'+'.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

#!/bin/sh		

dataset='office_home'
port=3011
GPUS=1
lr='5e-6'
cfg='configs/swin_base.yaml'
root='swin_base'

source='Clipart'
target='Art'
log_path="loghaha/${dataset}/${root}/${source}_${target}/${lr}"
out_path="resultshaha/${dataset}/${root}/${source}_${target}/${lr}"

python -m torch.distributed.run --nproc_per_node ${GPUS} --master_port ${port} dist_pmTrans.py --use-checkpoint \
--source ${source} --target ${target} --dataset ${dataset}  --tag PM --local_rank 0 --batch-size 32 --head_lr_ratio 10 --log ${log_path} --output ${out_path} \
--cfg ${cfg}




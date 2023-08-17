#!/bin/bash
#SBATCH --job-name=segtrain_1gpu    # Job name
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:2
#SBATCH --ntasks=2                  # same as num GPU
#SBATCH --cpus-per-task=15


CONFIG=$1
GPUS=2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

module load anaconda/2023a-pytorch

export PYTHONPATH="$(pwd)":$PYTHONPATH

# uncomment this if you get problems with distributed training
#export TORCH_DISTRIBUTED_DEBUG=INFO

python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(pwd)/tools/train.py \
    $CONFIG \
    --launcher pytorch ${@:2}

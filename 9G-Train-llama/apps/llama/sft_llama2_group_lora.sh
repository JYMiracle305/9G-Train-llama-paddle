#! /bin/bash

export MASTER_ADDR="localhost"
export MASTER_PORT=12345

CPM_PATH="/home/wanghanqing/projects/9G-Train" #改为自己的9G-Train仓库路径
CKPT_PATH=/home/wanghanqing/projects/models/Llama-2-7b-chat-cpm9g #改为自己的ckpt路径
EXP_PATH=/home/wanghanqing/projects/exp/LoRAs/group #改为自己的exp_path
MODEL_NAME="llama2-7b-chat-sft-group-lora-zh-code"

OPTS=""
OPTS+=" --model-config ${CKPT_PATH}/config.json"
OPTS+=" --tokenizer-path ${CKPT_PATH}"

OPTS+=" --train-iters 695"
OPTS+=" --inspect-iters 2000"
OPTS+=" --warmup-iters 20"

OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 0.1"
OPTS+=" --clip-grad 1.0"
OPTS+=" --loss-scale 1048576"
OPTS+=" --max-loss-scale 33554432"
OPTS+=" --min-loss-scale 1"
OPTS+=" --loss-scale-steps 32"

OPTS+=" --offload"
OPTS+=" --batch-size 8"
OPTS+=" --max-length 4096"
OPTS+=" --lr 1e-4"
OPTS+=" --start-step 0"
OPTS+=" --epoch 1"
OPTS+=" --load ${CKPT_PATH}/pytorch_model.pt"
OPTS+=" --dataset /home/wanghanqing/projects/exp/data/zh_and_code/bin_data_repeat" #改为自己的数据集路径
# TODO 这些 /data 在启元机器上需要改成 /home 下的路径
OPTS+=" --save ${EXP_PATH}"
OPTS+=" --save-name ${MODEL_NAME}"
OPTS+=" --tensorboard /home/wanghanqing/projects/exp/logs/tensorboard/${MODEL_NAME}/"
#lora超参数
OPTS+=" --delta-tuning"
OPTS+=" --delta-type lora"
OPTS+=" --lora-name zh"
OPTS+=" --lora-r 64"
OPTS+=" --lora-dropout 0.05"
OPTS+=" --lora-alpha 64"
OPTS+=" --lora-layer project_q project_v" #OPTS+=" --lora-layer project_q project_v project_k w_0 w_1 w_out"
OPTS+=" --save-origin-model"

#group_lora超参数
OPTS+=" --lora-root-path /home/wanghanqing/projects/exp/LoRAs"
#OPTs+=" --lora-save-path checkpoints"
OPTS+=" --lora-list zh-code"

OPTS+=" $@"


CMD="torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} ${CPM_PATH}/apps/llama/sft_llama_group_lora.py ${OPTS}" #使用自己的.py名称

echo "${CMD}"
$CMD
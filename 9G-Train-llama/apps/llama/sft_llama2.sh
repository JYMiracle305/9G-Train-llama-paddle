#! /bin/bash

set -x

export MASTER_ADDR="localhost"
export MASTER_PORT=12345

#CPM_PATH="/home/wangshuo1/code/9G-Train"
#CKPT_PATH=/data/public/opensource_models/meta-llama/Llama-2-7b-cpm9g
CPM_PATH="/home/songxin/9G-Train-llama"
CKPT_PATH=/home/songxin/9G-Train-llama/apps/llama/Llama-2-7b-cpm9g
EXP_PATH=.
MODEL_NAME="llama2-7b-sft"

OPTS=""
OPTS+=" --model-config ${CKPT_PATH}/config.json"
OPTS+=" --tokenizer-path ${CKPT_PATH}"

#OPTS+=" --train-iters 695"
OPTS+=" --inspect-iters 2000"
OPTS+=" --warmup-iters 20"
OPTS+=" --bf16"

OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 0.1"
OPTS+=" --clip-grad 1.0"
OPTS+=" --loss-scale 1048576"
OPTS+=" --max-loss-scale 33554432"
OPTS+=" --min-loss-scale 1"
OPTS+=" --loss-scale-steps 32"

OPTS+=" --offload"
OPTS+=" --batch-size 3"
OPTS+=" --max-length 4096"
OPTS+=" --lr 1e-4"
OPTS+=" --start-step 0"
OPTS+=" --epoch 8"
OPTS+=" --load ${CKPT_PATH}/pytorch_model.pt"
#OPTS+=" --dataset ${EXP_PATH}/resources/merge_qy_sft_bin"
OPTS+=" --dataset /home/songxin/9G-Train-llama/apps/llama/flan_plain_0809/data"
#OPTS+=" --dataset ./datasets.json"
#OPTS+=" --dataset ./bin_data_repeat"
# TODO 这些 /data 在启元机器上需要改成 /home 下的路径
OPTS+=" --save ${EXP_PATH}/checkpoints"
OPTS+=" --save-name ${MODEL_NAME}"
# OPTS+=" --save-model /data/models/${MODEL_NAME}/"
# OPTS+=" --tensorboard /data/logs/tensorboard/${MODEL_NAME}/${CUR_DATE}/"
# OPTS+=" --flash triton"
#OPTS+=" --flash cuda"
# OPTS+=" --load-grad"

OPTS+=" $@"


CMD="torchrun --nnodes=1 --nproc_per_node=16 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} ${CPM_PATH}/apps/llama/sft_llama.py ${OPTS}"
#CMD="torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} ${CPM_PATH}/apps/llama/pretrain_llama.py ${OPTS}"

echo "${CMD}"
$CMD 2>&1 | tee sft_train.3_4096_FA_bf16.log 

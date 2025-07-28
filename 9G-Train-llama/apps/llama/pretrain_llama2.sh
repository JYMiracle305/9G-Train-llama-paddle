#! /bin/bash

export MASTER_ADDR="localhost"
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)

SCRIPT_DIR="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
export PROJECT_ROOT="$(dirname "$SCRIPT_DIR")/.."
echo "当前目录为：{$PROJECT_ROOT}"

#CPM_PATH="./9G-Train"
#CKPT_PATH=/data/public/opensource_models/meta-llama/Llama-2-7b-cpm9g
CPM_PATH="${PROJECT_ROOT}"
CKPT_PATH="${PROJECT_ROOT}/data/metadata/Llama-2-7b-cpm9g"
EXP_PATH=.
MODEL_NAME="llama2-7b-train"

OPTS=""
OPTS+=" --model-config ${CKPT_PATH}/config.json"
OPTS+=" --tokenizer-path ${CKPT_PATH}"
OPTS+=" --vocab ./config/7b/vocab.txt"
OPTS+=" --train-iters 200"

OPTS+=" --max-length 4096"
OPTS+=" --inspect-iters 50"
OPTS+=" --lr-decay-style noam"
OPTS+=" --weight-decay 0.1"
OPTS+=" --clip-grad 1.0"
OPTS+=" --loss-scale 1048576"
OPTS+=" --loss-scale-steps 32"
#OPTS+=" --offload"
OPTS+=" --bf16"
OPTS+=" --flash cuda"
OPTS+=" --save ${EXP_PATH}/checkpoints"
OPTS+=" --save-name ${MODEL_NAME}"
OPTS+=" --save-model ${EXP_PATH}/models/llama-7b/"
OPTS+=" --log-dir ${EXP_PATH}/log"
OPTS+=" --tensorboard ${EXP_PATH}/log/tensorboard/"`date +"%Y%m%d%H%M%S"`
OPTS+=" --dataset ./datasets_bin_data_repeat.json"
#OPTS+=" --load ${CKPT_PATH}/pytorch_model.pt"
OPTS+=" --start-step 1"
OPTS+=" --save-iters 25000"

OPTS+=" --batch-size 4"
OPTS+=" --lr 1e-5"

OPTS+=" $@"

# export CUDA_VISIBLE_DEVICES=8,9
# CMD="torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} ${CPM_PATH}/apps/llama/pretrain_llama.py ${OPTS}"
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=./nccl_debug_%h_%p.log
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1

# export NCCL_HOME=/home/spack/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/nccl-2.18.3-1-glfr3mzrcsf2ginrtjsgvmluxlcknko5
# export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH
# export LD_PRELOAD=$NCCL_HOME/lib/libnccl.so.2
CMD="python -m paddle.distributed.launch \
        --nnodes=1 --nproc_per_node=2 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}
        --gpus=8,9  \
        ${CPM_PATH}/apps/llama/pretrain_llama.py ${OPTS}"
echo "${CMD}"
#nohup $CMD > incremental_train.4_4096_FA_bf16.lee.log 2>&1 &
$CMD 2>&1 | tee incremental_train.4_4096_FA_bf16.log 

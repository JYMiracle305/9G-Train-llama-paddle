#! /bin/bash

#export MASTER_ADDR=172.16.60.177
#export MASTER_PORT=12345

#CPM_PATH="./9G-Train"
#CKPT_PATH=/data/public/opensource_models/meta-llama/Llama-2-7b-cpm9g
CPM_PATH="/home/songxin/9G-Train-llama"
CKPT_PATH=/home/songxin/9G-Train-llama/apps/llama/Llama-2-7b-cpm9g
EXP_PATH=.
MODEL_NAME="llama2-7b-train"

OPTS=""
OPTS+=" --model-config ${CKPT_PATH}/config.json"
OPTS+=" --tokenizer-path ${CKPT_PATH}"
OPTS+=" --vocab ./config/7b/vocab.txt"

OPTS+=" --max-length 4096"
OPTS+=" --inspect-iters 100"
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
OPTS+=" --load ${CKPT_PATH}/pytorch_model.pt"
OPTS+=" --start-step 1"
OPTS+=" --save-iters 25000"
OPTS+=" --train-iters 5000"

OPTS+=" --batch-size 4"
OPTS+=" --lr 1e-5"

OPTS+=" $@"

GPUS_PER_NODE=16
MASTER_ADDR=172.16.60.177
MASTER_PORT=12088
NNODES=2
NODE_RANK=1 #work0 设置为1 ha

DDP_OPTIONS="--nproc_per_node $GPUS_PER_NODE \
             --nnodes $NNODES \
             --node_rank $NODE_RANK \
             --master_addr $MASTER_ADDR \
             --master_port $MASTER_PORT"

CMD="torchrun $DDP_OPTIONS ${CPM_PATH}/apps/llama/pretrain_llama.py ${OPTS}"

#CMD="torchrun --nnodes=2 --node_rank=0 --nproc_per_node=16 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} ${CPM_PATH}/apps/llama/pretrain_llama.py ${OPTS}"

echo "${CMD}"
$CMD  2>&1 | tee incremental_train.4_4096_FA_bf16_2nodes.log

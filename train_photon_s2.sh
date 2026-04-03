export UV_DEFAULT_INDEX="https://mirrors.aliyun.com/pypi/simple"
uv pip install nibabel qwen_vl_utils==0.0.11

apt-get update
apt install -y libibverbs1 ibverbs-providers ibverbs-utils

# NCCL
export NCCL_IB_GID_INDEX=3
export HF_ENDPOINT=https://hf-mirror.com

mkdir /tmp/cache
mkdir /tmp/cache/torch_temp
mkdir /tmp/cache/triton_temp

export PYTORCH_KERNEL_CACHE_PATH=/tmp/cache/torch_temp
export TRITON_CACHE_DIR=/tmp/cache/triton_temp

export FPS=1

# We use 8 GPU to train our model.

NNODES=$WORLD_SIZE \
NODE_RANK=$RANK \
MASTER_ADDR=$MASTER_ADDR \
MASTER_PORT=$MASTER_PORT \
NPROC_PER_NODE=$NPROC_PER_NODE \
FPS=1 \
swift sft \
    --model pretrained_weights/PhotonS1 \
    --model_type qwen2_5_vl \
    --template photon_s2 \
    --stage S2 \
    --dataset 'XXX' \
    --val_dataset 'XXX' \
    --load_from_cache_file true \
    --dataset_shuffle true \
    --train_dataloader_shuffle true \
    --torch_dtype bfloat16 \
    --train_type full \
    --attn_impl 'flash_attn' \
    --deepspeed 'zero2' \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --vit_lr 5e-6 \
    --save_strategy 'steps' \
    --eval_strategy 'no' \
    --save_steps 4000 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --output_dir output/ \
    --dataloader_num_workers 12 \
    --dataloader_prefetch_factor 12 \
    --warmup_ratio 0.05 \
    --freeze_vit false \
    --freeze_llm false \
    --freeze_aligner false \
    --gradient_checkpointing false \
    --vit_gradient_checkpointing false \

    

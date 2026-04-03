export UV_DEFAULT_INDEX="https://mirrors.aliyun.com/pypi/simple"
uv pip install nibabel qwen_vl_utils==0.0.11

apt-get update
apt install -y libibverbs1 ibverbs-providers ibverbs-utils

export NCCL_IB_GID_INDEX=3
export HF_ENDPOINT=https://hf-mirror.com

mkdir /tmp/cache
mkdir /tmp/cache/torch_temp
mkdir /tmp/cache/triton_temp

export PYTORCH_KERNEL_CACHE_PATH=/tmp/cache/torch_temp
export TRITON_CACHE_DIR=/tmp/cache/triton_temp
export FPS=1

# We use 16 GPU to train our model.

NNODES=$WORLD_SIZE \
NODE_RANK=$RANK \
MASTER_ADDR=$MASTER_ADDR \
MASTER_PORT=$MASTER_PORT \
NPROC_PER_NODE=$NPROC_PER_NODE \
FPS=1 \
swift sft \
    --model 'Modified-Qwen2.5-VL-3B-Instruct' \
    --model_type qwen2_5_vl \
    --template photon_s1 \
    --stage S1 \
    --dataset 'XXX' \
    --val_dataset 'XXX' \
    --dataset_shuffle true \
    --train_dataloader_shuffle true \
    --torch_dtype bfloat16 \
    --train_type full \
    --attn_impl 'flash_attn' \
    --deepspeed 'zero2' \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --learning_rate 1.25e-4 \
    --save_strategy 'steps' \
    --eval_strategy 'no' \
    --save_steps 2000 \
    --save_total_limit 4 \
    --logging_steps 5 \
    --output_dir output/Debug\
    --dataloader_num_workers 12 \
    --dataloader_prefetch_factor 12 \
    --freeze_parameters_ratio 1 \
    --trainable_parameters_regex ".*(vision_patch_embed).*" \
    --warmup_ratio 0.05 \
    --padding_free true \
    --gradient_checkpointing false \
    --vit_gradient_checkpointing false \
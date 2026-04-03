export UV_DEFAULT_INDEX="https://mirrors.aliyun.com/pypi/simple"

export NCCL_IB_GID_INDEX=3

apt-get update
apt install -y libibverbs1 ibverbs-providers ibverbs-utils
export FPS=1

NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MASTER_PORT=11223 \
FPS=1 \
swift infer \
    --model pretrained_weights/PhotonS1 \
    --infer_backend pt \
    --model_type qwen2_5_vl \
    --template photon_s1 \
    --stage S1 \
    --val_dataset 'XXX' \
    --attn_impl 'flash_attn' \
    --max_new_tokens 512 

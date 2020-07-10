export CUDA_VISIBLE_DEVICES=3

CKPT_DIR="./ckpt_ernie_pointwise_matching_tmp"

python -u ernie_pointwise_matching_predict.py \
            --batch_size=1 \
            --use_gpu=False \
            --checkpoint_dir=${CKPT_DIR} \
            --max_seq_len=128

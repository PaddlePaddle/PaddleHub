CKPT_DIR="./ckpt_ernie_pairwise_matching"

python -u ernie_pairwise_matching_predict.py \
            --batch_size=1 \
            --use_gpu=False \
            --checkpoint_dir=${CKPT_DIR} \
            --max_seq_len=128

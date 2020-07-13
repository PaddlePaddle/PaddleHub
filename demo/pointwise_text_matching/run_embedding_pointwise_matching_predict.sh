CKPT_DIR="./ckpt_embedding_pointwise_matching"

python -u embedding_pointwise_matching_predict.py \
            --batch_size=1 \
            --checkpoint_dir=${CKPT_DIR} \
            --max_seq_len=128 \
            --network=bow

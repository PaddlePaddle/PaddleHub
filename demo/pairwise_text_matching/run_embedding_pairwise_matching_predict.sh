CKPT_DIR="./ckpt_embedding_pairwise_matching"

python -u embedding_pairwise_matching_predict.py \
            --batch_size=1 \
            --checkpoint_dir=${CKPT_DIR} \
            --max_seq_len=128 \
            --network=bow

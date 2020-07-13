CKPT_DIR="./ckpt_embedding_pairwise_matching"

python -u embedding_pairwise_matching.py \
            --batch_size=128 \
            --checkpoint_dir=${CKPT_DIR} \
            --learning_rate=5e-2 \
            --max_seq_len=128 \
            --num_epoch=300 \
            --network=bow

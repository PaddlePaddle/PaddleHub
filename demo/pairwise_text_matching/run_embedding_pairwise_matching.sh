CKPT_DIR="./ckpt_embedding_pairwise_matching"

python -u embedding_pairwise_matching.py \
            --batch_size=32 \
            --checkpoint_dir=${CKPT_DIR} \
            --learning_rate=5e-5 \
            --max_seq_len=128 \
            --num_epoch=100 \
            --network=lstm

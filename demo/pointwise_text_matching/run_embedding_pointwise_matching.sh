export CUDA_VISIBLE_DEVICES=0
CKPT_DIR="./ckpt_embedding_pointwise_matching"

python -u embedding_pointwise_matching.py \
            --batch_size=128 \
            --checkpoint_dir=${CKPT_DIR} \
            --learning_rate=5e-3 \
            --max_seq_len=128 \
            --num_epoch=300 \
            --network=bow

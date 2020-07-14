CKPT_DIR="./ckpt_ernie_pointwise_matching"

python -u ernie_pointwise_matching.py \
            --batch_size=32 \
            --use_gpu=False \
            --checkpoint_dir=${CKPT_DIR} \
            --learning_rate=5e-5 \
            --max_seq_len=128 \
            --num_epoch=3 \
            --warmup_proportion=0.1 \
            --weight_decay=0.01 \
            --use_data_parallel=False \

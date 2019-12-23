export FLAGS_eager_delete_tensor_gb=0.0
export CUDA_VISIBLE_DEVICES=0

CKPT_DIR="./ckpt_stsb"

python -u regression.py \
                   --batch_size=32 \
                   --use_gpu=True \
                   --checkpoint_dir=${CKPT_DIR} \
                   --learning_rate=4e-5 \
                   --warmup_proportion=0.1 \
                   --weight_decay=0.1 \
                   --max_seq_len=128 \
                   --num_epoch=3 \
                   --use_data_parallel=False

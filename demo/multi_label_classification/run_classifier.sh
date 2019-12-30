export FLAGS_eager_delete_tensor_gb=0.0
export CUDA_VISIBLE_DEVICES=0

CKPT_DIR="./ckpt_toxic"

python -u multi_label_classifier.py \
                   --batch_size=32 \
                   --use_gpu=True \
                   --checkpoint_dir=${CKPT_DIR} \
                   --learning_rate=5e-5 \
                   --weight_decay=0.01 \
                   --max_seq_len=128 \
                   --warmup_proportion=0.1 \
                   --num_epoch=3

export CUDA_VISIBLE_DEVICES=3

CKPT_DIR="./ckpt"

python -u finetune_with_hub.py \
                   --batch_size 16 \
                   --weight_decay  0.01 \
                   --checkpoint_dir $CKPT_DIR \
                   --num_epoch 3 \
                   --max_seq_len 256 \
                   --learning_rate 5e-5

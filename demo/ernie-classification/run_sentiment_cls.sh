export CUDA_VISIBLE_DEVICES=5

CKPT_DIR="./ckpt_sentiment_cls"
python -u sentiment_cls.py \
                   --batch_size 32 \
                   --use_gpu=False \
                   --weight_decay  0.01 \
                   --checkpoint_dir $CKPT_DIR \
                   --num_epoch 3 \
                   --max_seq_len 128 \
                   --learning_rate 5e-5

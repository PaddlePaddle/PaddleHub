export CUDA_VISIBLE_DEVICES=3

CKPT_DIR="./ckpt_dbqa"
python -u question_answering.py \
                   --batch_size 8 \
                   --weight_decay  0.01 \
                   --checkpoint_dir $CKPT_DIR \
                   --num_epoch 3 \
                   --max_seq_len 512 \
                   --learning_rate 2e-5

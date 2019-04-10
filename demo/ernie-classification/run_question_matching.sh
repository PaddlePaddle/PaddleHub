export CUDA_VISIBLE_DEVICES=0

CKPT_DIR="./ckpt_question_matching"
python -u question_matching.py \
                   --batch_size 32 \
                   --weight_decay  0.0 \
                   --checkpoint_dir $CKPT_DIR \
                   --num_epoch 3 \
                   --max_seq_len 128 \
                   --learning_rate 2e-5

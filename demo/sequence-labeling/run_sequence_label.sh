export CUDA_VISIBLE_DEVICES=0

CKPT_DIR="./ckpt_sequence_label"
python -u sequence_label.py \
                   --batch_size 16 \
                   --weight_decay  0.01 \
                   --num_epoch 3 \
                   --checkpoint_dir $CKPT_DIR \
                   --max_seq_len 256 \
                   --learning_rate 5e-5 \
                   --use_pyreader True \
                   --use_data_parallel True

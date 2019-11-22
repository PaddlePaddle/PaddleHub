export FLAGS_eager_delete_tensor_gb=0.0

CKPT_DIR="./ckpt_sequence_label"
python -u sequence_label.py \
                   --batch_size 16 \
                   --weight_decay  0.01 \
                   --num_epoch 5 \
                   --checkpoint_dir $CKPT_DIR \
                   --max_seq_len 256 \
                   --learning_rate 2e-5 \
                   --use_pyreader True \
                   --use_data_parallel True

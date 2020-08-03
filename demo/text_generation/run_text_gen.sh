export FLAGS_eager_delete_tensor_gb=0.0
export CUDA_VISIBLE_DEVICES=0

CKPT_DIR="./ckpt_generation"
python -u text_gen.py \
                   --batch_size 16 \
                   --num_epoch 30 \
                   --checkpoint_dir $CKPT_DIR \
                   --max_seq_len 50 \
                   --learning_rate 5e-3 \
                   --cut_fraction 0.1 \
                   --use_data_parallel True \
                   --use_gpu True

export FLAGS_eager_delete_tensor_gb=0.0
export CUDA_VISIBLE_DEVICES=2,3

python -u reading_comprehension.py \
                   --batch_size=12 \
                   --use_gpu=True \
                   --checkpoint_dir="./ckpt_rc" \
                   --learning_rate=3e-5 \
                   --weight_decay=0.01 \
                   --warmup_proportion 0.1 \
                   --num_epoch=2 \
                   --max_seq_len 384 \
                   --use_pyreader=True \
                   --use_data_parallel=True \
                   --version_2_with_negative=False

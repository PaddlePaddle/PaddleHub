export FLAGS_eager_delete_tensor_gb=0.0

dataset=squad
python -u reading_comprehension.py \
                   --batch_size=8 \
                   --use_gpu=True \
                   --checkpoint_dir=./ckpt_${dataset} \
                   --learning_rate=3e-5 \
                   --weight_decay=0.01 \
                   --warmup_proportion=0.1 \
                   --num_epoch=2 \
                   --max_seq_len=384 \
                   --use_pyreader=True \
                   --use_data_parallel=True \
                   --dataset=${dataset}

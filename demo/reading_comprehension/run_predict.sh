export FLAGS_eager_delete_tensor_gb=0.0
export CUDA_VISIBLE_DEVICES=0

python -u  predict.py \
                   --batch_size=1 \
                   --use_gpu=True \
                   --checkpoint_dir="./ckpt_squad" \
                   --max_seq_len=512 \

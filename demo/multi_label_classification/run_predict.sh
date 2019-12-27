export FLAGS_eager_delete_tensor_gb=0.0
export CUDA_VISIBLE_DEVICES=0

CKPT_DIR="./ckpt_toxic"
python -u predict.py --checkpoint_dir $CKPT_DIR --max_seq_len 128 --use_gpu True

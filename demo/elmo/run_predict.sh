export FLAGS_eager_delete_tensor_gb=0.0
export CUDA_VISIBLE_DEVICES=0

CKPT_DIR="./ckpt_chnsenticorp"
python -u predict.py --checkpoint_dir $CKPT_DIR  --use_gpu True

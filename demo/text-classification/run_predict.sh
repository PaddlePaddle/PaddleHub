export CUDA_VISIBLE_DEVICES=0

CKPT_DIR="./ckpt_20190414203357/best_model"
python -u predict.py --checkpoint_dir $CKPT_DIR --max_seq_len 128

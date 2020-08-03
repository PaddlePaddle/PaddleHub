export CUDA_VISIBLE_DEVICES=0

CKPT_DIR="./ckpt_chnsenticorp "
python -u predict.py --checkpoint_dir $CKPT_DIR --use_gpu True --max_seq_len=96

export CUDA_VISIBLE_DEVICES=5

CKPT_DIR="./ckpt_sentiment_cls/best_model"
python -u cls_predict.py --checkpoint_dir $CKPT_DIR --max_seq_len 128

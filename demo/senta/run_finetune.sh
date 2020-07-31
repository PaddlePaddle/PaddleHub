export CUDA_VISIBLE_DEVICES=1,2,3

CKPT_DIR="./ckpt_chnsenticorp"

python -u senta_finetune.py \
                   --batch_size=24 \
                   --max_seq_len=96 \
                   --use_gpu=True \
                   --checkpoint_dir=${CKPT_DIR} \
                   --num_epoch=3

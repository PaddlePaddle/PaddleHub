export CUDA_VISIBLE_DEVICES=0

DATASET="chnsenticorp"
CKPT_DIR="./ckpt_${DATASET}"

python -u senta_finetune.py \
                   --batch_size=24 \
                   --use_gpu=False \
                   --checkpoint_dir=${CKPT_DIR} \
                   --num_epoch=3

export FLAGS_eager_delete_tensor_gb=0.0
export CUDA_VISIBLE_DEVICES=0

CKPT_DIR="./ckpt_chnsenticorp"

python -u senta_finetune.py \
                   --batch_size=24 \
                   --use_gpu=True \
                   --checkpoint_dir=${CKPT_DIR} \
                   --num_epoch=3

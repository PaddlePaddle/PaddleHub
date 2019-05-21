export CUDA_VISIBLE_DEVICES=0

python -u elmo_finetune.py \
                   --batch_size=32 \
                   --use_gpu=True \
                   --checkpoint_dir="./ckpt_chnsenticorp" \
                   --learning_rate=1e-4 \
                   --weight_decay=1 \
                   --num_epoch=3

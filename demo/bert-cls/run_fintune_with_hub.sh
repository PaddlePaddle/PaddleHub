export CUDA_VISIBLE_DEVICES=2

DATA_PATH=./chnsenticorp_data

rm -rf $CKPT_PATH
python -u finetune_with_hub.py \
                   --use_cuda true \
                   --batch_size 32 \
                   --in_tokens false \
                   --data_dir ${DATA_PATH} \
                   --weight_decay  0.01 \
                   --warmup_proportion 0.0 \
                   --validation_steps 50 \
                   --epoch 3 \
                   --max_seq_len 128 \
                   --learning_rate 5e-5 \
                   --skip_steps 10

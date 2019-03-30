export CUDA_VISIBLE_DEVICES=5

DATA_PATH=./chnsenticorp_data

rm -rf ./bert_cls_ckpt
python -u finetune_with_hub.py \
                   --batch_size 32 \
                   --in_tokens false \
                   --data_dir ${DATA_PATH} \
                   --weight_decay  0.01 \
                   --warmup_proportion 0.0 \
                   --epoch 3 \
                   --max_seq_len 128 \
                   --learning_rate 5e-5

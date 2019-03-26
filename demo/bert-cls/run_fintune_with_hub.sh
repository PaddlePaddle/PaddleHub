export CUDA_VISIBLE_DEVICES=6

BERT_BASE_PATH="chinese_L-12_H-768_A-12"
TASK_NAME='chnsenticorp'
DATA_PATH=chnsenticorp_data

rm -rf $CKPT_PATH
python -u finetune_with_hub.py \
                   --use_cuda true \
                   --batch_size 4096 \
                   --in_tokens true \
                   --data_dir ${DATA_PATH} \
                   --vocab_path ${BERT_BASE_PATH}/vocab.txt \
                   --weight_decay  0.01 \
                   --warmup_proportion 0.0 \
                   --validation_steps 50 \
                   --epoch 3 \
                   --max_seq_len 128 \
                   --learning_rate 5e-5 \
                   --skip_steps 10

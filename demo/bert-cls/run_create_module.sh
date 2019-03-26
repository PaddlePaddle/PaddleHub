export FLAGS_enable_parallel_graph=1
export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=0

BERT_BASE_PATH="chinese_L-12_H-768_A-12"
TASK_NAME='chnsenticorp'
DATA_PATH=chnsenticorp_data
CKPT_PATH=chn_checkpoints

python -u create_module.py --task_name ${TASK_NAME} \
                   --use_cuda true \
                   --do_train true \
                   --do_val true \
                   --do_test true \
                   --batch_size 4096 \
                   --in_tokens true \
                   --init_pretraining_params ${BERT_BASE_PATH}/params \
                   --data_dir ${DATA_PATH} \
                   --vocab_path ${BERT_BASE_PATH}/vocab.txt \
                   --checkpoints ${CKPT_PATH} \
                   --save_steps 100 \
                   --weight_decay  0.01 \
                   --warmup_proportion 0.0 \
                   --validation_steps 50 \
                   --epoch 3 \
                   --max_seq_len 128 \
                   --bert_config_path ${BERT_BASE_PATH}/bert_config.json \
                   --learning_rate 5e-5 \
                   --skip_steps 10

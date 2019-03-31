export CUDA_VISIBLE_DEVICES=5

DATA_PATH=./chnsenticorp_data


HUB_MODULE_DIR="./hub_module/bert_chinese_L-12_H-768_A-12.hub_module"
#HUB_MODULE_DIR="./hub_module/ernie_stable.hub_module"
CKPT_DIR="./ckpt"
#rm -rf $CKPT_DIR
python -u finetune_with_hub.py \
                   --batch_size 128 \
                   --hub_module_dir=$HUB_MODULE_DIR \
                   --data_dir ${DATA_PATH} \
                   --weight_decay  0.01 \
                   --checkpoint_dir $CKPT_DIR \
                   --warmup_proportion 0.0 \
                   --epoch 2 \
                   --max_seq_len 16 \
                   --learning_rate 5e-5

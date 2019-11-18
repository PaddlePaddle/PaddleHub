export FLAGS_eager_delete_tensor_gb=0.0
export CUDA_VISIBLE_DEVICES=0

DATASET="express_ner"
CKPT_DIR="./ckpt_${DATASET}"

python -u sequence_label.py \
        --batch_size 32 \
        --dataset=${DATASET} \
        --weight_decay  0.01 \
        --num_epoch 3 \
        --checkpoint_dir $CKPT_DIR \
        --max_seq_len 128 \
        --learning_rate 5e-5 \
        --use_pyreader True \
        --use_data_parallel True \
        --add_crf True

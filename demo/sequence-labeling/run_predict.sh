export FLAGS_eager_delete_tensor_gb=0.0
export CUDA_VISIBLE_DEVICES=0

DATASET="express_ner"
CKPT_DIR="./ckpt_${DATASET}"

python -u predict.py \
        --checkpoint_dir $CKPT_DIR \
        --max_seq_len 128 \
        --use_gpu True \
        --dataset=${DATASET} \
        --add_crf=True

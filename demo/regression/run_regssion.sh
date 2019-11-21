export FLAGS_eager_delete_tensor_gb=0.0
export CUDA_VISIBLE_DEVICES=0

DATASET="STS-B"
CKPT_DIR="./ckpt_${DATASET}"
# Recommending hyper parameters for difference task
# STS-B: batch_size=32, weight_decay=0.1, num_epoch=3, max_seq_len=128, lr=4e-5

python -u regression.py \
                   --batch_size=32 \
                   --use_gpu=True \
                   --dataset=${DATASET} \
                   --checkpoint_dir=${CKPT_DIR} \
                   --learning_rate=4e-5 \
                   --weight_decay=0.1 \
                   --max_seq_len=128 \
                   --num_epoch=3 \
                   --use_pyreader=True \
                   --use_data_parallel=True

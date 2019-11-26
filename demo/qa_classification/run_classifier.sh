export FLAGS_eager_delete_tensor_gb=0.0
export CUDA_VISIBLE_DEVICES=0

CKPT_DIR="./ckpt_qa"
# Recommending hyper parameters for difference task
# ChnSentiCorp: batch_size=24, weight_decay=0.01, num_epoch=3, max_seq_len=128, lr=5e-5
# NLPCC_DBQA: batch_size=8, weight_decay=0.01, num_epoch=3, max_seq_len=512, lr=2e-5
# LCQMC: batch_size=32, weight_decay=0, num_epoch=3, max_seq_len=128, lr=2e-5

python -u classifier.py \
                   --batch_size=24 \
                   --use_gpu=True \
                   --checkpoint_dir=${CKPT_DIR} \
                   --learning_rate=5e-5 \
                   --weight_decay=0.01 \
                   --max_seq_len=128 \
                   --num_epoch=3 \
                   --use_pyreader=False \
                   --use_data_parallel=False \

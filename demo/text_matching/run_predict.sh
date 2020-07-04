export FLAGS_eager_delete_tensor_gb=0.0
export CUDA_VISIBLE_DEVICES=0

CKPT_DIR="./ckpt_text_matching"

python -u predict.py \
            --batch_size=1 \
            --use_gpu=True \
            --checkpoint_dir=${CKPT_DIR} \
            --learning_rate=5e-5 \
            --max_seq_len=256 \
            --num_epoch=3 \
            --use_data_parallel=False \
            --is_pair_wise=True \
            --network=lstm

export FLAGS_eager_delete_tensor_gb=0.0
export CUDA_VISIBLE_DEVICES=0

CKPT_DIR="./ckpt_cmrc2018"
dataset=cmrc2018

python -u  predict.py \
                   --batch_size=8 \
                   --use_gpu=True \
                   --dataset=${dataset} \
                   --checkpoint_dir=${CKPT_DIR} \
                   --learning_rate=2.5e-5 \
                   --weight_decay=0.01 \
                   --warmup_proportion=0.1 \
                   --num_epoch=1 \
                   --max_seq_len=512 \
                   --use_pyreader=False \
                   --use_data_parallel=False

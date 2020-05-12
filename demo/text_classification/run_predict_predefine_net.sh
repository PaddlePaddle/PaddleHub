export FLAGS_eager_delete_tensor_gb=0.0
export CUDA_VISIBLE_DEVICES=0

CKPT_DIR="./ckpt_chnsenticorp_predefine_net"

python -u predict_predefine_net.py \
                    --checkpoint_dir=$CKPT_DIR \
                    --max_seq_len=128 \
                    --use_gpu=True \
                    --batch_size=24 \
                    --network=bilstm

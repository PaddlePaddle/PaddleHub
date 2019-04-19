export CUDA_VISIBLE_DEVICES=0

# User can select chnsenticorp, nlpcc_dbqa, lcqmc for different task
DATASET="chnsenticorp"
CKPT_DIR="./ckpt_${DATASET}"

python -u text_classifier.py \
                   --batch_size=24 \
                   --use_gpu=False \
                   --checkpoint_dir=${CKPT_DIR} \
                   --num_epoch=10

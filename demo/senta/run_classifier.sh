export CUDA_VISIBLE_DEVICES=2

# User can select chnsenticorp, nlpcc_dbqa, lcqmc for different task
DATASET="chnsenticorp"
CKPT_DIR="./ckpt_${DATASET}"

python -u text_classifier.py \
                   --batch_size=24 \
                   --use_gpu=True \
                   --checkpoint_dir=${CKPT_DIR} \
                   --num_epoch=10

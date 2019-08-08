export FLAGS_eager_delete_tensor_gb=0.0
export CUDA_VISIBLE_DEVICES=0

# User can select chnsenticorp, nlpcc_dbqa, lcqmc and so on for different task
DATASET="chnsenticorp"
CKPT_DIR="./ckpt_${DATASET}"

# Recommending hyper parameters for difference task
# ChnSentiCorp: batch_size=24, weight_decay=0.01, num_epoch=3, max_seq_len=128, lr=5e-5
# NLPCC_DBQA: batch_size=8, weight_decay=0.01, num_epoch=3, max_seq_len=512, lr=2e-5
# LCQMC: batch_size=32, weight_decay=0, num_epoch=3, max_seq_len=128, lr=2e-5
# MNLI: batch_size=32, weight_decay=0, num_epoch=3, max_seq_len=128, lr=5e-5
# QQP: batch_size=32, weight_decay=0, num_epoch=3, max_seq_len=128, lr=5e-5
# QNLI: batch_size=32, weight_decay=0, num_epoch=3, max_seq_len=128, lr=5e-5
# SST-2: batch_size=32, weight_decay=0, num_epoch=3, max_seq_len=128, lr=5e-5
# CoLA: batch_size=32, weight_decay=0, num_epoch=3, max_seq_len=128, lr=5e-5
# MRPC: batch_size=32, weight_decay=0, num_epoch=3, max_seq_len=128, lr=5e-5
# RTE: batch_size=32, weight_decay=0, num_epoch=3, max_seq_len=128, lr=3e-5
# XNLI: batch_size=32, weight_decay=0, num_epoch=2, max_seq_len=128, lr=5e-5
#       Specify the language with an underscore like xnli_zh.
#       ar- Arabic      bg- Bulgarian      de- German
#       el- Greek       en- English        es- Spanish
#       fr- French      hi- Hindi          ru- Russian
#       sw- Swahili     th- Thai           tr- Turkish
#       ur- Urdu        vi- Vietnamese     zh- Chinese (Simplified)

python -u text_classifier.py \
                   --batch_size=32 \
                   --use_gpu=True \
                   --dataset=${DATASET} \
                   --checkpoint_dir=${CKPT_DIR} \
                   --learning_rate=5e-5 \
                   --weight_decay=0.01 \
                   --max_seq_len=128 \
                   --num_epoch=3 \
                   --use_pyreader=True \
                   --use_data_parallel=True \
                   --use_taskid=False \

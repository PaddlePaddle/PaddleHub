export FLAGS_eager_delete_tensor_gb=0.0
export CUDA_VISIBLE_DEVICES=0

# User can select chnsenticorp, nlpcc_dbqa, lcqmc and so on for different task
DATASET="chnsenticorp"
CKPT_DIR="./ckpt_${DATASET}"
# Recommending hyper parameters for difference task
# ChnSentiCorp: batch_size=24, weight_decay=0.01, num_epoch=3, max_seq_len=128, lr=5e-5
# NLPCC_DBQA: batch_size=8, weight_decay=0.01, num_epoch=3, max_seq_len=512, lr=2e-5
# LCQMC: batch_size=32, weight_decay=0, num_epoch=3, max_seq_len=128, lr=2e-5
# MRPC: Testing
# QQP: Testing
# SST-2: Testing
# CoLA: Testing
# QNLI: Testing
# RTE: Testing
# MNLI: Testing
# XNLI: Specify the language with an underscore like xnli_zh.
#       ar- Arabic      bg- Bulgarian      de- German
#       el- Greek       en- English        es- Spanish
#       fr- French      hi- Hindi          ru- Russian
#       sw- Swahili     th- Thai           tr- Turkish
#       ur- Urdu        vi- Vietnamese     zh- Chinese (Simplified)

python -u predict.py --checkpoint_dir $CKPT_DIR --max_seq_len 50 --use_gpu False --dataset=${DATASET}

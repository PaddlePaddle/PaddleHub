export FLAGS_eager_delete_tensor_gb=0.0
export CUDA_VISIBLE_DEVICES=0

# User can select chnsenticorp, nlpcc_dbqa, lcqmc and so on for different task
DATASET="chnsenticorp"
CKPT_DIR="./ckpt_${DATASET}"

python -u text_classifier.py \
                   --batch_size=24 \
                   --use_gpu=True \
                   --dataset=${DATASET} \
                   --checkpoint_dir=${CKPT_DIR} \
                   --learning_rate=5e-5 \
                   --weight_decay=0.01 \
                   --max_seq_len=128 \
                   --num_epoch=3 \
                   --use_pyreader=True \
                   --use_data_parallel=True

# Recommending hyper parameters for difference task
# for ChineseGLUE:
# TNews: batch_size=32, weight_decay=0, num_epoch=3, max_seq_len=128, lr=5e-5
# LCQMC: batch_size=32, weight_decay=0, num_epoch=3, max_seq_len=128, lr=5e-5
# XNLI_zh: batch_size=32, weight_decay=0, num_epoch=2, max_seq_len=128, lr=5e-5
# INEWS: batch_size=4, weight_decay=0, num_epoch=3, max_seq_len=512, lr=5e-5
# DRCD: see demo: reading-comprehension
# CMRC2018: see demo: reading-comprehension
# BQ: batch_size=32, weight_decay=0, num_epoch=2, max_seq_len=100, lr=1e-5
# MSRANER: see demo: sequence-labeling
# THUCNEWS: batch_size=8, weight_decay=0, num_epoch=2, max_seq_len=512, lr=5e-5
# IFLYTEKDATA: batch_size=16, weight_decay=0, num_epoch=5, max_seq_len=256, lr=1e-5

# for other tasks:
# ChnSentiCorp: batch_size=24, weight_decay=0.01, num_epoch=3, max_seq_len=128, lr=5e-5
# NLPCC_DBQA: batch_size=8, weight_decay=0.01, num_epoch=3, max_seq_len=512, lr=2e-5
# LCQMC: batch_size=32, weight_decay=0, num_epoch=3, max_seq_len=128, lr=2e-5
# QQP: batch_size=32, weight_decay=0, num_epoch=3, max_seq_len=128, lr=5e-5
# QNLI: batch_size=32, weight_decay=0, num_epoch=3, max_seq_len=128, lr=5e-5
# SST-2: batch_size=32, weight_decay=0, num_epoch=3, max_seq_len=128, lr=5e-5
# CoLA: batch_size=32, weight_decay=0, num_epoch=3, max_seq_len=128, lr=5e-5
# MRPC: batch_size=32, weight_decay=0.01, num_epoch=3, max_seq_len=128, lr=5e-5
# RTE: batch_size=32, weight_decay=0, num_epoch=3, max_seq_len=128, lr=3e-5
# MNLI: batch_size=32, weight_decay=0, num_epoch=3, max_seq_len=128, lr=5e-5
#       Specify the matched/mismatched dev and test dataset  with an underscore.
#       mnli_m or mnli: dev and test in matched dataset.
#       mnli_mm: dev and test in mismatched dataset.
#      The difference can be seen in https://www.nyu.edu/projects/bowman/multinli/paper.pdf.
#       If you are not sure which one to pick, just use mnli or mnli_m.
# XNLI: batch_size=32, weight_decay=0, num_epoch=3, max_seq_len=128, lr=5e-5
#       Specify the language with an underscore like xnli_zh.
#       ar- Arabic      bg- Bulgarian      de- German
#       el- Greek       en- English        es- Spanish
#       fr- French      hi- Hindi          ru- Russian
#       sw- Swahili     th- Thai           tr- Turkish
#       ur- Urdu        vi- Vietnamese     zh- Chinese (Simplified)

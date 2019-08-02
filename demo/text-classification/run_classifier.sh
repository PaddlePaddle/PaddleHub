export FLAGS_eager_delete_tensor_gb=0.0
export CUDA_VISIBLE_DEVICES=0

# User can select chnsenticorp, nlpcc_dbqa, lcqmc and so on for different task

DATASET="chnsenticorp"
CKPT_DIR="./ckpt_${DATASET}"

# Support ChnSentiCorp	NLPCC_DBQA	LCQMC	MRPC	QQP	SST-2
#         CoLA	QNLI	RTE	MNLI	XNLI
# for XNLI: Specify the language with an underscore like xnli_zh.
#       ar: Arabic      bg: Bulgarian      de: German
#       el: Greek       en: English        es: Spanish
#       fr: French      hi: Hindi          ru: Russian
#       sw: Swahili     th: Thai           tr: Turkish
#       ur: Urdu        vi: Vietnamese     zh: Chinese (Simplified)

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
                   --use_data_parallel=True \
                   --use_taskid=False \

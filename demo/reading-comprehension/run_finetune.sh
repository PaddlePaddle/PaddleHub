export FLAGS_eager_delete_tensor_gb=0.0
export CUDA_VISIBLE_DEVICES=0

# Recommending hyper parameters for difference task
# squad: batch_size=8, weight_decay=0, num_epoch=3, max_seq_len=512, lr=5e-5
# squad2.0: batch_size=8, weight_decay=0, num_epoch=3, max_seq_len=512, lr=5e-5
# cmrc2018: batch_size=8, weight_decay=0, num_epoch=2, max_seq_len=512, lr=2.5e-5
# drcd: batch_size=8, weight_decay=0, num_epoch=2, max_seq_len=512, lr=2.5e-5

dataset=cmrc2018
python -u reading_comprehension.py \
                   --batch_size=8 \
                   --use_gpu=True \
                   --checkpoint_dir=./ckpt_${dataset} \
                   --learning_rate=2.5e-5 \
                   --weight_decay=0.01 \
                   --warmup_proportion=0.1 \
                   --num_epoch=2 \
                   --max_seq_len=512 \
                   --dataset=${dataset}

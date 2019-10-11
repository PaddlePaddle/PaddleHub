OUTPUT=result/

hub autofinetune img_cls.py \
    --param_file=hparam.yaml \
    --cuda=['6'] \
    --popsize=5 \
    --round=10 \
    --output_dir=${OUTPUT} \
    --evaluate_choice=fulltrail \
    --tuning_strategy=pshe2

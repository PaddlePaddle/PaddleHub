OUTPUT=result

hub autofinetune img_cls.py \
    --param_file=hparam.yaml \
    --gpu=0 \
    --popsize=15 \
    --round=10 \
    --output_dir=${OUTPUT} \
    --evaluator=fulltrail \
    --tuning_strategy=pshe2

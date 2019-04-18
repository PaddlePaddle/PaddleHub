cuda_visible_devices=0
module=resnet50
num_epoch=1
batch_size=16
use_gpu=False
checkpoint_dir=paddlehub_finetune_ckpt

while getopts "gm:n:b:c:d:" options
do
	case "$options" in
        m)
            module=$OPTARG;;
        n)
            num_epoch=$OPTARG;;
        b)
            batch_size=$OPTARG;;
        c)
            checkpoint_dir=$OPTARG;;
        d)
            cuda_visible_devices=$OPTARG;;
        g)
            use_gpu=True;;
		?)
			echo "unknown options"
            exit 1;;
	esac
done

export CUDA_VISIBLE_DEVICES=${cuda_visible_devices}

python -u img_classifier.py --target finetune --use_gpu ${use_gpu} --batch_size ${batch_size} --checkpoint_dir ${checkpoint_dir} --num_epoch ${num_epoch} --module ${module}

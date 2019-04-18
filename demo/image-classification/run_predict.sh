cuda_visible_devices=0
module=resnet50
use_gpu=False
checkpoint_dir=paddlehub_finetune_ckpt

while getopts "gm:c:d:" options
do
	case "$options" in
        m)
            module=$OPTARG;;
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

python -u predict.py --use_gpu ${use_gpu} --checkpoint_dir ${checkpoint_dir} --module ${module}

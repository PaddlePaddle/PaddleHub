# coding:utf-8
import argparse
import os
import ast
import shutil

import paddle.fluid as fluid
import paddlehub as hub
from paddlehub.common.logger import logger

parser = argparse.ArgumentParser(__doc__)
parser.add_argument(
    "--epochs", type=int, default=5, help="Number of epoches for fine-tuning.")
parser.add_argument(
    "--use_gpu",
    type=ast.literal_eval,
    default=True,
    help="Whether use GPU for fine-tuning.")
parser.add_argument(
    "--checkpoint_dir", type=str, default=None, help="Path to save log data.")

# the name of hyperparameters to be searched should keep with hparam.py
parser.add_argument(
    "--batch_size",
    type=int,
    default=16,
    help="Total examples' number in batch for training.")
parser.add_argument(
    "--learning_rate", type=float, default=1e-4, help="learning_rate.")

# saved_params_dir and model_path are needed by auto finetune
parser.add_argument(
    "--saved_params_dir",
    type=str,
    default="",
    help="Directory for saving model")
parser.add_argument(
    "--model_path", type=str, default="", help="load model path")


def is_path_valid(path):
    if path == "":
        return False
    path = os.path.abspath(path)
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    return True


def finetune(args):
    # Load Paddlehub resnet50 pretrained model
    module = hub.Module(name="mobilenet_v2_imagenet")
    input_dict, output_dict, program = module.context(trainable=True)

    # Download dataset and use ImageClassificationReader to read dataset
    dataset = hub.dataset.Flowers()
    data_reader = hub.reader.ImageClassificationReader(
        image_width=module.get_expected_image_width(),
        image_height=module.get_expected_image_height(),
        images_mean=module.get_pretrained_images_mean(),
        images_std=module.get_pretrained_images_std(),
        dataset=dataset)

    feature_map = output_dict["feature_map"]

    img = input_dict["image"]
    feed_list = [img.name]

    # Select finetune strategy, setup config and finetune
    strategy = hub.DefaultFinetuneStrategy(learning_rate=args.learning_rate)

    config = hub.RunConfig(
        use_cuda=True,
        num_epoch=args.epochs,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        strategy=strategy)

    # Construct transfer learning network
    task = hub.ImageClassifierTask(
        data_reader=data_reader,
        feed_list=feed_list,
        feature=feature_map,
        num_classes=dataset.num_labels,
        config=config)

    # Load model from the defined model path or not
    if args.model_path != "":
        with task.phase_guard(phase="train"):
            task.init_if_necessary()
            task.load_parameters(args.model_path)
            logger.info("PaddleHub has loaded model from %s" % args.model_path)

    task.finetune()
    run_states = task.eval()
    eval_avg_score, eval_avg_loss, eval_run_speed = task._calculate_metrics(
        run_states)

    # Move ckpt/best_model to the defined saved parameters directory
    best_model_dir = os.path.join(config.checkpoint_dir, "best_model")
    if is_path_valid(args.saved_params_dir) and os.path.exists(best_model_dir):
        shutil.copytree(best_model_dir, args.saved_params_dir)
        shutil.rmtree(config.checkpoint_dir)

    # acc on dev will be used by auto finetune
    print("AutoFinetuneEval" + "\t" + str(float(eval_avg_score["acc"])))


if __name__ == "__main__":
    args = parser.parse_args()
    finetune(args)

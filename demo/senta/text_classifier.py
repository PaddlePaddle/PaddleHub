import argparse
import ast

import paddle.fluid as fluid
import paddlehub as hub

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=3, help="Number of epoches for fine-tuning.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=False, help="Whether use GPU for finetuning, input should be True or False")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--batch_size", type=int, default=32, help="Total examples' number in batch for training.")
args = parser.parse_args()
# yapf: enable.

if __name__ == '__main__':
    # Step1: load Paddlehub senta pretrained model
    module = hub.Module(name="senta")
    inputs, outputs, program = module.context(trainable=True)

    # Step2: Download dataset and use TextClassificationReader to read dataset
    dataset = hub.dataset.ChnSentiCorp()

    reader = hub.reader.LACTokenizeReader(
        dataset=dataset, vocab_path=module.get_vocab_path())

    sent_feature = outputs["sequence_output"]

    # Define a classfication finetune task by PaddleHub's API
    cls_task = hub.create_text_cls_task(
        feature=sent_feature, num_classes=dataset.num_labels)

    # Setup feed list for data feeder
    # Must feed all the tensor of senta's module need
    feed_list = [inputs["words"].name, cls_task.variable('label').name]

    strategy = hub.finetune.strategy.AdamWeightDecayStrategy(
        learning_rate=1e-3, weight_decay=0.01, warmup_proportion=0.01)

    config = hub.RunConfig(
        use_cuda=args.use_gpu,
        num_epoch=args.num_epoch,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        strategy=strategy)

    # Finetune and evaluate by PaddleHub's API
    # will finish training, evaluation, testing, save model automatically
    hub.finetune_and_eval(
        task=cls_task, data_reader=reader, feed_list=feed_list, config=config)

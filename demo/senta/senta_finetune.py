# coding:utf-8
import argparse
import ast

import paddle.fluid as fluid
import paddlehub as hub

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=3, help="Number of epoches for fine-tuning.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=True, help="Whether use GPU for fine-tuning, input should be True or False")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--max_seq_len", type=int, default=128, help="Number of words of the longest seqence.")
parser.add_argument("--batch_size", type=int, default=32, help="Total examples' number in batch for training.")
args = parser.parse_args()
# yapf: enable.

jieba_paddle = hub.Module(name='jieba_paddle')


def cut(text):
    res = jieba_paddle.cut(text, use_paddle=False)
    return res


if __name__ == '__main__':
    # Load Paddlehub senta pretrained model
    module = hub.Module(name="senta_bow", version='1.2.0')
    inputs, outputs, program = module.context(
        trainable=True, max_seq_len=args.max_seq_len)

    # Tokenizer tokenizes the text data and encodes the data as model needed.
    # If you use transformer modules (ernie, bert, roberta and so on), tokenizer should be hub.BertTokenizer.
    # Otherwise, tokenizer should be hub.CustomTokenizer.
    # If you choose CustomTokenizer, you can also change the chinese word segmentation tool, for example jieba.
    tokenizer = hub.CustomTokenizer(
        vocab_file=module.get_vocab_path(),
        tokenize_chinese_chars=True,
        cut_function=cut,  # jieba.cut as cut function
    )

    dataset = hub.dataset.ChnSentiCorp(
        tokenizer=tokenizer, max_seq_len=args.max_seq_len)

    # Construct transfer learning network
    # Use sentence-level output.
    sent_feature = outputs["sentence_feature"]

    # Setup RunConfig for PaddleHub Fine-tune API
    config = hub.RunConfig(
        use_cuda=args.use_gpu,
        num_epoch=args.num_epoch,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        strategy=hub.AdamWeightDecayStrategy())

    # Define a classfication fine-tune task by PaddleHub's API
    cls_task = hub.TextClassifierTask(
        dataset=dataset,
        feature=sent_feature,
        num_classes=dataset.num_labels,
        config=config)
    cls_task.finetune_and_eval()

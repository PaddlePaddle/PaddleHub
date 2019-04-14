import paddle.fluid as fluid
import paddlehub as hub

# Step1: Select pre-trained model
module = hub.Module(name="ernie")
inputs, outputs, program = module.context(trainable=True, max_seq_len=128)

# Step2: Prepare Dataset and DataReader
dataset = hub.dataset.ChnSentiCorp()
reader = hub.reader.ClassifyReader(
    dataset=dataset, vocab_path=module.get_vocab_path(), max_seq_len=128)

# Step3: Construct transfer learning task
with fluid.program_guard(program):
    label = fluid.layers.data(name="label", shape=[1], dtype='int64')
    pooled_output = outputs["pooled_output"]
    cls_task = hub.create_text_cls_task(
        feature=pooled_output, label=label, num_classes=dataset.num_labels)

# Step4: Setup config then start finetune
strategy = hub.AdamWeightDecayStrategy(learning_rate=5e-5, weight_decay=0.01)

config = hub.RunConfig(
    use_cuda=True,
    checkpoint_dir="./ckpt",
    num_epoch=3,
    batch_size=32,
    strategy=strategy)

feed_list = [
    inputs["input_ids"].name, inputs["position_ids"].name,
    inputs["segment_ids"].name, inputs["input_mask"].name, label.name
]

hub.finetune_and_eval(
    task=cls_task, data_reader=reader, feed_list=feed_list, config=config)

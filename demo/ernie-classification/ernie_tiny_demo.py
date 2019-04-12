import paddle.fluid as fluid
import paddlehub as hub

module = hub.Module(name="ernie")
inputs, outputs, program = module.context(trainable=True, max_seq_len=128)

reader = hub.reader.ClassifyReader(
    dataset=hub.dataset.ChnSentiCorp(),
    vocab_path=module.get_vocab_path(),
    max_seq_len=128)

with fluid.program_guard(program):
    label = fluid.layers.data(name="label", shape=[1], dtype='int64')

    pooled_output = outputs["pooled_output"]

    feed_list = [
        inputs["input_ids"].name, inputs["position_ids"].name,
        inputs["segment_ids"].name, inputs["input_mask"].name, label.name
    ]

    cls_task = hub.create_text_classification_task(
        pooled_output, label, num_classes=reader.get_num_labels())

    strategy = hub.BERTFinetuneStrategy(
        weight_decay=0.01,
        learning_rate=5e-5,
        warmup_strategy="linear_warmup_decay",
    )

    config = hub.RunConfig(
        use_cuda=True, num_epoch=3, batch_size=32, strategy=strategy)

    hub.finetune_and_eval(
        task=cls_task, data_reader=reader, feed_list=feed_list, config=config)

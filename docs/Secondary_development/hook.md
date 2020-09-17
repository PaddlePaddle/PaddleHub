# 如何修改Task内置方法？


了解如何修改Task内置方法，我们首先了解下Task中的事件。

Task定义了[组网事件](./how_to_define_task.md)和[运行事件](./how_to_define_task.md)。其中运行事件的工作流程如下图。

![](../imgs/task_event_workflow.png)


**NOTE:**
* 图中提到的运行设置config参见[RunConfig说明](../reference/config.md)
* "finetune_start_event"，"finetune_end_event"，"predict_start_event"，"predict_end_event"，
"eval_start_event"，"eval_end_event"等事件是用于打印相应阶段的日志信息。"save_ckpt_interval_event"事件用于保存当前训练的模型参数。"log_interval_event"事件用于计算模型评价指标以及可视化这些指标。

如果您需要对图中提到的事件的具体实现进行修改，可以通过Task提供的事件回调hook机制进行改写。如下示例中将PaddleHub log_interval_event默认的accuracy评价指标改为F1评价指标:

```python
import time
from collections import OrderedDict

import numpy as np
import paddlehub as hub


def calculate_f1_np(preds, labels):
    # 计算F1分数
    # preds：预测label
    # labels： 真实labels
    # 返回F1分数
    preds = np.array(preds)
    labels = np.array(labels)
    tp = np.sum((labels == 1) & (preds == 1))
    tn = np.sum((labels == 0) & (preds == 0))
    fp = np.sum((labels == 0) & (preds == 1))
    fn = np.sum((labels == 1) & (preds == 0))
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f1 = (2 * p * r) / (p + r) if p + r else 0
    return f1

# 自定义评估方法实现
def calculate_metrics(self, run_states):
    # run_states: list类型，每个元素是一个RunState对象，指明了该step的运行状态
    # 返回评估得分，平均损失值和平局运行速度
    loss_sum = acc_sum = run_examples = 0
    run_step = run_time_used = 0
    all_labels = np.array([])
    all_infers = np.array([])

    for run_state in run_states:
        run_examples += run_state.run_examples
        run_step += run_state.run_step
        loss_sum += np.mean(
            run_state.run_results[-1]) * run_state.run_examples
        acc_sum += np.mean(
            run_state.run_results[2]) * run_state.run_examples
        np_labels = run_state.run_results[0]
        np_infers = run_state.run_results[1]
        all_labels = np.hstack((all_labels, np_labels.reshape([-1])))
        all_infers = np.hstack((all_infers, np_infers.reshape([-1])))

    run_time_used = time.time() - run_states[0].run_time_begin
    avg_loss = loss_sum / run_examples
    run_speed = run_step / run_time_used

    scores = OrderedDict()
    f1 = calculate_f1_np(all_infers, all_labels)
    scores["f1"] = f1

    return scores, avg_loss, run_speed


# 改写_log_interval_event实现
def new_log_interval_event(self, run_states):
    # 改写的事件方法，参数列表务必与PaddleHub内置的相应方法保持一致
    scores, avg_loss, run_speed = calculate_metrics(self, run_states)
    formatted_scores = ", ".join(["%s: %.5f"%(key, value) for key, value in scores.items()])
    print("[new_log_interval_event] step %d / %d: loss=%.5f %s[step/sec: %.2f]" %
            (self.current_step, self.max_train_steps, avg_loss,
                formatted_scores, run_speed))


# 最简单的PaddleHub运行样例
module = hub.Module(name="ernie_tiny")
inputs, outputs, program = module.context(
    trainable=True, max_seq_len=128)
tokenizer = hub.ErnieTinyTokenizer(
            vocab_file=module.get_vocab_path(),
            spm_path=module.get_spm_path(),
            word_dict_path=module.get_word_dict_path())
dataset = hub.dataset.ChnSentiCorp(
        tokenizer=tokenizer, max_seq_len=128)
task = hub.TextClassifierTask(
    dataset=dataset,
    feature=outputs["pooled_output"],
    num_classes=dataset.num_labels,
    )

# 利用Hook改写PaddleHub内置_log_interval_event实现，需要2步(假设task已经创建好)
# 1.删除PaddleHub内置_log_interval_event实现
# hook_type：你想要改写的事件hook类型
# name：hook名字，“default”表示PaddleHub内置_log_interval_event实现
task.delete_hook(hook_type="log_interval_event", name="default")

# 2.增加自定义_log_interval_event实现(new_log_interval_event)
# hook_type：你想要改写的事件hook类型
# name: hook名字
# func：自定义改写的方法
task.add_hook(hook_type="log_interval_event", name="new_log_interval_event", func=new_log_interval_event)

# 输出hook信息
print(task.hooks_info())

task.finetune_and_eval()
```

**NOTE:**
* 关于上述提到的run_states参见[RunEnv说明](../reference/task/runenv.md)
* 改写的事件方法，参数列表务必与PaddleHub内置的相应方法保持一致。
* 只支持改写/删除以下事件hook类型：
 "build_env_start_event"，"build_env_end_event"，"finetune_start_event"，"finetune_end_event"，
 "predict_start_event"，"predict_end_event"，"eval_start_event"，"eval_end_event"，
 "log_interval_event"，"save_ckpt_interval_event"，"eval_interval_event"，"run_step_event"。
* 如果想要改写组网事件，Hook不支持。改写组网事件参见[自定义Task](./how_to_define_task.md)。
* 如何创建Task，参见[PaddleHub迁移学习示例](../../demo)

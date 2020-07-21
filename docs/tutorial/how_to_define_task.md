# 自定义Task

本节内容讲述如何实现自定义Task。在了解本节内容之前，您需要先了解以下内容：
* 任务基类[BasicTask](../reference/task/base_task.md)
* 运行状态[RunState](../reference/task/runstate.md)
* 运行环境[RunEnv](../reference/task/runenv.md)

当自定义一个Task时，我们并不需要重新实现eval、finetune等通用接口。一般来讲，新的Task与其他Task的区别在于
* 网络结构
* 评估指标

这两者的差异可以通过重载BasicTask的组网事件和运行事件来实现

## 组网事件
BasicTask定义了一系列的组网事件，当需要构建对应的Fluid Program时，相应的事件会被调用。通过重载实现对应的组网函数，用户可以实现自定义网络

###  `_build_net`
进行前向网络组网的函数，用户需要自定义实现该函数，函数需要返回对应预测结果的Variable list

```python
# 代码示例
def _build_net(self):
    cls_feats = self.feature
    if self.hidden_units is not None:
        for n_hidden in self.hidden_units:
            cls_feats = fluid.layers.fc(
                input=cls_feats, size=n_hidden, act="relu")

    logits = fluid.layers.fc(
        input=cls_feats,
        size=self.num_classes,
        param_attr=fluid.ParamAttr(
            name="cls_out_w",
            initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
        bias_attr=fluid.ParamAttr(
            name="cls_out_b", initializer=fluid.initializer.Constant(0.)),
        act="softmax")

    return [logits]
```
###  `_add_label`
添加label的函数，用户需要自定义实现该函数，函数需要返回对应输入label的Variable list

```python
# 代码示例
def _add_label(self):
    return [fluid.layers.data(name="label", dtype="int64", shape=[1])]
```

###  `_add_metrics`
添加度量指标的函数，用户需要自定义实现该函数，函数需要返回对应度量指标的Variable list

```python
# 代码示例
def _add_metrics(self):
    return [fluid.layers.accuracy(input=self.outputs[0], label=self.label)]
```

## 运行事件
BasicTask定义了一系列的运行时回调事件，在特定的时机时触发对应的事件，在自定的Task中，通过重载实现对应的回调函数，用户可以实现所需的功能

###  `_build_env_start_event`

当需要进行一个新的运行环境构建时，该事件被触发。通过重载实现该函数，用户可以在一个环境开始构建前进行对应操作，例如写日志

```python
# 代码示例
def _build_env_start_event(self):
    logger.info("Start to build env {}".format(self.phase))
```

###  `_build_env_end_event`
当一个新的运行环境构建完成时，该事件被触发。通过继承实现该函数，用户可以在一个环境构建结束后进行对应操作，例如写日志

```python
# 代码示例
def _build_env_end_event(self):
    logger.info("End of build env {}".format(self.phase))
```
###  `_finetune_start_event`
当开始一次finetune时，该事件被触发。通过继承实现该函数，用户可以在开始一次finetune操作前进行对应操作，例如写日志

```python
# 代码示例
def _finetune_start_event(self):
    logger.info("PaddleHub finetune start")
```

###  `_finetune_end_event`
当结束一次finetune时，该事件被触发。通过继承实现该函数，用户可以在结束一次finetune操作后进行对应操作，例如写日志

```python
# 代码示例
def _finetune_end_event(self):
    logger.info("PaddleHub finetune finished.")
```

###  `_eval_start_event`
当开始一次evaluate时，该事件被触发。通过继承实现该函数，用户可以在开始一次evaluate操作前进行对应操作，例如写日志

```python
# 代码示例
def _eval_start_event(self):
    logger.info("Evaluation on {} dataset start".format(self.phase))
```
###  `_eval_end_event`
当结束一次evaluate时，该事件被触发。通过继承实现该函数，用户可以在完成一次evaluate操作后进行对应操作，例如计算运行速度、评估指标等

```python
# 代码示例
def _eval_end_event(self, run_states):
    run_step = 0
    for run_state in run_states:
        run_step += run_state.run_step

    run_time_used = time.time() - run_states[0].run_time_begin
    run_speed = run_step / run_time_used
    logger.info("[%s dataset evaluation result] [step/sec: %.2f]" %
        (self.phase, run_speed))
```
* `run_states`: 一个list对象，list中的每一个元素都是RunState对象，该list包含了整个评估过程的状态数据。

###  `_predict_start_event`
当开始一次predict时，该事件被触发。通过继承实现该函数，用户可以在开始一次predict操作前进行对应操作，例如写日志

```python
# 代码示例
def _predict_start_event(self):
    logger.info("PaddleHub predict start")
```

###  `_predict_end_event`
当结束一次predict时，该事件被触发。通过继承实现该函数，用户可以在结束一次predict操作后进行对应操作，例如写日志

```python
# 代码示例
def _predict_end_event(self):
    logger.info("PaddleHub predict finished.")
```

###  `_log_interval_event`
调用*finetune* 或者 *finetune_and_eval*接口时，每当命中用户设置的日志打印周期时（[RunConfig.log_interval](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub-API:-RunConfig#log_interval)）。通过继承实现该函数，用户可以在finetune过程中定期打印所需数据，例如计算运行速度、loss、准确率等

```python
# 代码示例
def _log_interval_event(self, run_states):
    avg_loss, avg_acc, run_speed = self._calculate_metrics(run_states)
    self.env.loss_scalar.add_record(self.current_step, avg_loss)
    self.env.acc_scalar.add_record(self.current_step, avg_acc)
    logger.info("step %d: loss=%.5f acc=%.5f [step/sec: %.2f]" %
        (self.current_step, avg_loss, avg_acc, run_speed))
```
* `run_states`: 一个list对象，list中的每一个元素都是RunState对象，该list包含了整个从上一次该事件被触发到本次被触发的状态数据

###  `_save_ckpt_interval_event`
调用*finetune* 或者 *finetune_and_eval*接口时，每当命中用户设置的保存周期时（[RunConfig.save_ckpt_interval](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub-API:-RunConfig#save_ckpt_interval)），该事件被触发。通过继承实现该函数，用户可以在定期保存checkpoint

```python
# 代码示例
def _save_ckpt_interval_event(self):
    self.save_checkpoint(self.current_epoch, self.current_step)
```

###  `_eval_interval_event`
调用*finetune_and_eval*接口时，每当命中用户设置的评估周期时（[RunConfig.eval_interval](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub-API:-RunConfig#eval_interval)），该事件被触发。通过继承实现该函数，用户可以实现自定义的评估指标计算

```python
# 代码示例
def _eval_interval_event(self):
    self.eval(phase="dev")
```

###  `_run_step_event`
调用*eval*、*predict*、*finetune_and_eval*、*finetune*等接口时，每执行一次计算，该事件被触发。通过继承实现该函数，用户可以实现所需操作

```python
# 代码示例
def _run_step_event(self, run_state):
    ...
```
* `run_state`: 一个RunState对象，指明了该step的运行状态

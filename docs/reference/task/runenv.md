# Class `hub.task.RunEnv`
PaddleHub的Task通过phase来切换不同的状态（训练/评估/预测等），每个phase都有对应的一个RunEnv对象用于保存对应状态下的重要属性。对于不需要自定义Task的用户来说，本文档并不是必须的

## Property `current_step`
该环境所执行过的step数

## Property `current_epoch`
该环境所执行过的epoch数量

## Property `main_program`
该环境对应的主Program，包含训练/评估/预测所需的计算图

## Property `startup_program`
该环境对应的初始化Program，包含初始化操作的计算图

## Property `py_reader`
该环境中的PyReader对象

## Property `reader`
该环境下对应的数据Reader

## Property `loss`
该环境下对应的loss Variable

## Property `label`
该环境下对应的label Variable

## Property `outputs`
该环境下对应的outputs Variable

## Property `metrics`
该环境下对应的metrics Variable，只在test和train phase有效

## Property  `is_inititalized `
该环境是否已经完成初始化

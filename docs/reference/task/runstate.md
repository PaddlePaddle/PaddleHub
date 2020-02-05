# Class `hub.task.RunState`
PaddleHub的Task在进行训练/评估/预测时，会将运行过程中的状态和输出保存到RunState对象中。用户可以从RunState对象中获取所需数据

## Property `run_time_begin`
运行的开始时间

## Property `run_step`
运行的step数

## Property `run_examples`
运行的样本数量

## Property `run_results`
运行的结果列表，数据和task中的fetch_list一一对应

## Property `run_time_used`
运行所用时间

## Property `run_speed`
运行速度

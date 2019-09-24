# PaddleHub 超参优化（Auto Fine-tune）

## 一、简介

机器学习训练模型的过程中自然少不了调参。模型的参数可分成两类：参数与超参数，前者是模型通过自身的训练学习得到的参数数据；后者则需要通过人工经验设置（如学习率、dropout_rate、batch_size等），以提高模型训练的效果。当前模型往往参数空间大，手动调参十分耗时，尝试成本高。PaddleHub  Auto Fine-tune可以实现自动调整超参数。

PaddleHub Auto Fine-tune提供两种超参优化策略：

* HAZero: 核心思想是通过对正态分布中协方差矩阵的调整来处理变量之间的依赖关系和scaling。算法基本可以分成以下三步: 采样产生新解；计算目标函数值；更新正态分布参数。调整参数的基本思路为，调整参数使得产生更优解的概率逐渐增大

* PSHE2: 采用粒子群算法，最优超参数组合就是所求问题的解。现在想求得最优解就是要找到更新超参数组合，即如何更新超参数，才能让算法更快更好的收敛到最优解。PSE2算法根据超参数本身历史的最优，在一定随机扰动的情况下决定下一步的更新方向。

PaddleHub Auto Fine-tune提供两种超参评估策略：

* FullTrail: 给定一组超参，利用这组超参从头开始Finetune一个新模型，之后在数据集dev部分评估这个模型

* ModelBased: 给定一组超参，若这组超参来自第一轮优化的超参，则从头开始Finetune一个新模型；若这组超参数不是来自第一轮优化的超参数，则程序会加载前几轮已经Fine-tune完毕后保存的较好模型，基于这个模型，在当前的超参数组合下继续Finetune。这个Fine-tune完毕后保存的较好模型，评估方式是这个模型在数据集dev部分的效果。

## 二、准备工作

使用PaddleHub Auto Fine-tune必须准备两个文件，并且这两个文件需要按照指定的格式书写。这两个文件分别是需要Fine-tune的python脚本finetuee.py和需要优化的超参数信息yaml文件hparam.yaml。

[PaddleHub Auto Fine-tune超参优化--NLP情感分类任务](https://github.com/PaddlePaddle/PaddleHub/blob/release/v1.2/tutorial/autofinetune-nlp.md)

[PaddleHub Auto Fine-tune超参优化--CV图像分类任务](https://github.com/PaddlePaddle/PaddleHub/blob/release/v1.2/tutorial/autofinetune-cv.md)

## 三、启动方式

**确认安装PaddleHub版本在1.2.0以上, 同时PaddleHub Auto Fine-tune功能要求至少有一张GPU显卡可用。**

通过以下命令方式：
```shell
$ OUTPUT=result/
$ hub autofinetune finetunee.py --param_file=hparam.yaml --cuda=['1','2'] --popsize=5 --round=10
 --output_dir=${OUTPUT} --evaluate_choice=fulltrail --tuning_strategy=pshe2
```

其中，选项

> `--param_file`: 需要优化的超参数信息yaml文件

> `--cuda`: 设置运行程序的可用GPU卡号，list类型，中间以逗号隔开，不能有空格，默认为[‘0’]

> `--popsize`: 设置程序运行每轮产生的超参组合数，默认为5

> `--round`: 设置程序运行的轮数，默认是10

> `--output_dir`: 设置程序运行输出结果存放目录，可选，不指定该选项参数时，在当前运行路径下生成存放程序运行输出信息的文件夹

> `--evaluate_choice`: 设置自动优化超参的评价效果方式，可选fulltrail和modelbased, 默认为fulltrail

> `--tuning_strategy`: 设置自动优化超参策略，可选hazero和pshe2，默认为hazero

**NOTE:** Auto Fine-tune功能会根据popsize和cuda自动实现排队使用GPU，如popsize=5，cuda=['0','1','2','3']，则每搜索一轮，Auto Fine-tune自动起四个进程训练，所以第5组超参组合需要排队一次。为了提高GPU利用率以及超参优化效率，此时建议可以设置为3张可用的卡，cuda=['0','1','2']。


## 四、可视化

Auto Finetune API在优化超参过程中会自动对关键训练指标进行打点，启动程序后执行下面命令

```shell
$ tensorboard --logdir $OUTPUT/tb_paddle --host ${HOST_IP} --port ${PORT_NUM}
```

其中${HOST_IP}为本机IP地址，${PORT_NUM}为可用端口号，如本机IP地址为192.168.0.1，端口号8040，
用浏览器打开192.168.0.1:8040，即可看到搜素过程中各超参以及指标的变化情况

## 五、其他

1. 如在使用Auto Fine-tune功能时，输出信息中包含如下字样：

**WARNING：Program which was ran with hyperparameters as ... was crashed!**

首先根据终端上的输出信息，确定这个输出信息是在第几个round（如round 3），之后查看${OUTPUT}/round3/下的日志文件信息log.info, 查看具体出错原因。

2. PaddleHub AutoFinetune 命令行支持从启动命令hub autofinetune传入finetunee.py中不需要搜索的选项参数，如上述示例中的max_seq_len选项，可以参照以下方式传入。

```shell
$ OUTPUT=result/
$ hub autofinetune finetunee.py --param_file=hparam.yaml --cuda=['1','2'] --popsize=5 --round=10
 --output_dir=${OUTPUT} --evaluate_choice=fulltrail --tuning_strategy=pshe2 max_seq_len 128
```

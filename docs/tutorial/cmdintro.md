# PaddleHub命令行工具

PaddleHub为预训练模型的管理和使用提供了命令行工具。

PaddleHub支持修改预训练模型存放路径：
* 如已设置`${HUB_HOME}`环境变量，则预训练模型、配置等文件都存放在`${HUB_HOME}`指示的路径下
* 如未设置`${HUB_HOME}`环境变量，则存放在`$HOME`指示的路径下

目前命令行支持以下12个命令：

## `install`

用于将Module安装到本地，默认安装在`${HUB_HOME}/.paddlehub/modules`目录下，当一个Module安装到本地后，用户可以通过其他命令操作该Module（例如，使用该Module进行预测），也可以使用PaddleHub提供的python API，将Module应用到自己的任务中，实现迁移学习

## `uninstall`

用于卸载本地Module

## `show`

用于查看本地已安装Module的属性或者指定目录下确定的Module的属性，包括其名字、版本、描述、作者等信息

## `download`

用于下载百度提供的Module

`选项`
* `--output_path`：用于指定存放下载文件的目录，默认为当前目录

* `--uncompress`：是否对下载的压缩包进行解压，默认不解压

* `--type`：指定下载的资源类型，当指定Model时，download只会下载Model的资源。默认为All，此时会优先搜索Module资源，如果没有相关的Module资源，则搜索Model

## `search`

通过关键字在服务端检索匹配的Module，当想要查找某个特定模型的Module时，使用search命令可以快速得到结果，例如`hub search ssd`命令，会查找所有包含了ssd字样的Module，命令支持正则表达式，例如`hub search ^s.*`搜索所有以s开头的资源。

`注意`
如果想要搜索全部的Module，使用`hub search *`并不生效，这是因为shell会自行进行通配符展开，将*替换为当前目录下的文件名。为了进行全局搜索，用户可以直接键入`hub search`

## `list`

列出本地已经安装的Module

## `run`

用于执行Module的预测，需要注意的是，并不是所有的模型都支持预测（同样，也不是所有的模型都支持迁移学习），更多关于run命令的细节，请查看下方的`关于预测`

## `help`

显示帮助信息

## `version`

显示PaddleHub版本信息

## `clear`

PaddleHub在使用过程中会产生一些缓存数据，这部分数据默认存放在${HUB_HOME}/.paddlehub/cache目录下，用户可以通过clear命令来清空缓存

## `autofinetune`

用于自动调整Fine-tune任务的超参数，具体使用详情参考[PaddleHub AutoDL Finetuner使用教程](https://github.com/PaddlePaddle/PaddleHub/blob/release/v1.4/tutorial/autofinetune.md)

`选项`
* `--param_file`: 需要搜索的超参数信息yaml文件

* `--gpu`: 设置运行程序的可用GPU卡号，中间以逗号隔开，不能有空格

* `--popsize`: 设置程序运行每轮产生的超参组合数，默认为5

* `--round`: 设置程序运行的轮数，默认是10

* `--output_dir`: 设置程序运行输出结果存放目录，可选，不指定该选项参数时，在当前运行路径下生成存放程序运行输出信息的文件夹

* `--evaluator`: 设置自动搜索超参的评价效果方式，可选fulltrail和populationbased, 默认为populationbased

* `--strategy`: 设置自动搜索超参算法，可选hazero和pshe2，默认为hazero


## `config`
用于查看和设置paddlehub相关设置，包括对server地址、日志级别的设置

`示例`
* `hub config`: 显示当前paddlehub的设置

* `hub config reset`: 恢复当前paddlehub的设置为默认设置

* `hub config server==[address]`: 设置当前server地址为[address]

* `hub config log==[level]`: 设置当前日志级别为[level]， 可选值为critical, error, warning, info, debug, nolog, 从左到右优先级从高到低，nolog表示不显示日志信息

## `serving`

用于一键部署Module预测服务，详细用法见[PaddleHub Serving一键服务部署](serving.md)

**NOTE:**

1. 在PaddleHub中，Module表示一个`可执行的神经网络模型`，一个Module可以支持直接命令行预测，也可以配合PaddleHub Finetune API，通过少量代码实现迁移学习。不是所有的Module都支持命令行预测 (例如BERT/ERNIE Transformer类模型，一般需要搭配任务进行finetune)，也不是所有的Module都可用于finetune（例如LAC词法分析模型，我们不建议用户用于finetune）。

PaddleHub尽量简化了用户在使用命令行预测时的理解成本，一般来讲，我们将预测分为NLP和CV两大类

## NLP类的任务
输入数据通过--input_text或者--input_file指定。以百度LAC模型（中文词法分析）为例，可以通过以下两个命令实现单行文本和多行文本的分析。

```shell
# 单文本预测
$ hub run lac --input_text "今天是个好日子"
```
```shell
# 多文本分析
$ hub run lac --input_file test.txt
```

其中test.txt的样例格式如下，每行是一个需要词法分析句子

```
今天是个好日子
天气预报说今天要下雨
下一班地铁马上就要到了
……更多行……
```

## CV类的任务
输入数据通过`--input_path`或者`--input_file`指定。以SSD模型（单阶段目标检测）为例子，可以通过以下两个命令实现单张图片和多张图片的预测

```shell
# 单张照片预测
$ hub run ssd_mobilenet_v1_pascal --input_path test.jpg
```
```shell
# 多张照片预测
$ hub run ssd_mobilenet_v1_pascal --input_file test.txt
```
其中test.txt的格式为
```
cat.jpg
dog.jpg
person.jpg
……更多行……
```

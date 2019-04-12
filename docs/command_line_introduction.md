# PaddleHub 命令行介绍

PaddleHub为Module/Model（关于Model和Module的区别，请查看下方的介绍）的管理和使用提供了命令行工具，目前命令行支持以下10个命令：

## `install`

用于将Module安装到本地，默认安装在`${USER_HOME}/.paddlehub/module`目录下，当一个Module安装到本地后，用户可以通过其他命令操作该Module（例如，使用该Module进行预测），也可以使用PaddleHub提供的python API，将Module应用到自己的任务中，实现迁移学习

## `uninstall`

用于卸载本地Module

## `show`

用于查看Module的属性，包括Module的名字、版本、描述、作者等信息

## `download`

用于下载百度提供的预训练Model

`选项`
> `--output_path`：用于指定存放下载文件的目录，默认为当前目录
>
> `--uncompress`：是否对下载的压缩包进行解压，默认不解压

## `search`

通过关键字在服务端检索匹配的Module/Model，当想要查找某个特定模型的Module/Model时，使用search命令可以快速得到结果，例如`hub search ssd`命令，会查找所有包含了ssd字样的Module/Model，命令支持正则表达式，例如`hub search ^s.*`搜索所有以s开头的资源。

`注意`
如果想要搜索全部的Module/Model，使用`hub search *`并不生效，这是因为shell会自行进行通配符展开，将*替换为当前目录下的文件名。为了进行全局搜索，用户可以直接键入`hub search`

## `list`

列出本地已经安装的Module

## `run`

用于执行Module的预测，需要注意的是，并不是所有的模型都支持预测（同样，也不是所有的模型都支持迁移学习），更多关于run命令的细节，请查看下方的`关于预测`

## `help`

显示帮助信息

## `version`

显示版本信息

## `clear`

PaddleHub在使用过程中会产生一些缓存数据，这部分数据默认存放在${USER_HOME}/.paddlehub/cache目录下，用户可以通过clear命令来清空缓存

# 关于预测
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
$ hub run ssd_mobilenet_pascal --input_path test.jpg
```
```shell
# 多张照片预测
$ hub run ssd_mobilenet_pascal --input_file test.txt
```
其中test.txt的格式为
```
cat.jpg
dog.jpg
person.jpg
……更多行……
```

# 关于Model和Module

在PaddleHub中，我们明确区分开Module/Model两个概念

## Model

Model表示预训练好的参数和模型，当需要使用Model进行预测时，需要模型配套的代码，进行模型的加载，数据的预处理等操作后，才能进行预测。

PaddleHub为PaddlePaddle生态的预训练模型提供了统一的管理机制，用户可以使用`hub download`命令的获取到最新的Model，以便进行实验或者其他操作。

## Module

Module是Model的超集，是一个`可执行模块`，一个Module可以支持直接命令行预测，也可以配合PaddleHub Finetune API，通过少量代码实现迁移学习。
需要注意的是，不是所有的Module都支持命令行预测; (例如BERT/ERNIE Transformer类模型，一般需要搭配任务进行finetune)
也不是所有的Module都可用于finetune（例如LAC词法分析模型，我们不建议用户用于finetune）。

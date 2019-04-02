# 命令行
Paddle Hub为Module的管理和使用提供了命令行工具，目前命令行支持以下9个命令：

* `install`：用于将Module安装到本地，默认安装在${USER_HOME}/.hub/module目录下，当一个Module安装到本地后，用户可以通过其他命令操作该Module（例如，使用该Module进行预测），也可以使用PaddleHub提供的python API，将Module应用到自己的任务中，实现迁移学习

* `uninstall`：用于卸载本地Module

* `show`：用于查看Module的属性，包括Module的名字、版本、描述、作者等信息

* `download`：用于下载百度NLP工具包

* `search`：通过关键字在服务端检索匹配的Module，当你想要查找某个特定模型的Module时，使用search命令可以快速得到结果，例如`hub search ssd`命令，会查找所有包含了ssd字样的Module

* `list`：列出本地已经安装的Module

* `run`：用于执行Module的预测，需要注意的是，并不是所有的模型都支持预测（同样，也不是所有的模型都支持迁移学习）

* `help`：显示帮助信息

* `version`：显示版本信息

# 关于预测
PaddleHub尽量简化了用户在使用命令行预测时的理解成本，一般来讲，我们将预测分为NLP和CV两大类

## NLP类的任务
输入数据通过--input_text或者--input_file指定。以LAC（中文词性分析）为例子，可以通过以下两个命令实现单文本和多文本的预测
```shell
#单文本预测
hub run lac --input_text "今天是个好日子"
```
```shell
#多文本分析
hub run lac --input_file test.csv
```
其中test.csv的格式为
```
今天是个好日子
天气预报说今天要下雨
下一班地铁马上就要到了
……更多行……
```
## CV类的任务
输入数据通过--input_path或者--input_file指定。以SSD（目标检测）为例子，可以通过以下两个命令实现单张图片和多张图片的预测
```shell
#单张照片预测
hub run ssd_mobilenet_pascal --input_text test.jpg
```
```shell
#多文本分析
hub run ssd_mobilenet_pascal --input_file test.csv
```
其中test.csv的格式为
```
cat.jpg
dog.jpg
person.jpg
……更多行……
```

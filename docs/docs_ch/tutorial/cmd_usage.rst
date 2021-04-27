===========================
PaddleHub命令行工具
===========================

PaddleHub为预训练模型的管理和使用提供了命令行工具。

我们一共提供了11个命令，涵盖了模型安装、卸载、预测等等各方面。

hub install
==================

用于将Module安装到本地，默认安装在`${HUB_HOME}/.paddlehub/modules`目录下，当一个Module安装到本地后，用户可以通过其他命令操作该Module（例如，使用该Module进行预测），也可以使用PaddleHub提供的python API，将Module应用到自己的任务中，实现迁移学习

.. tip::

    如果设置了环境变量 *${HUB_HOME}* ，则预训练模型和相关的配置文件都会保存到指定的 *${HUB_HOME}* 路径下。
    如果未设置环境变量 *${HUB_HOME}* ，则预训练模型和相关的配置文件都会保存到用户的主目录 *$HOME* 下。

hub uninstall
==================

用于卸载本地Module

hub show
==================

用于查看本地已安装Module的属性或者指定目录下确定的Module的属性，包括其名字、版本、描述、作者等信息

hub download
==================

用于下载PaddleHub提供的Module

hub search
==================

通过关键字在服务端检索匹配的Module，当想要查找某个特定模型的Module时，使用search命令可以快速得到结果，例如`hub search ssd`命令，会查找所有包含了ssd字样的Module，命令支持正则表达式，例如`hub search ^s.*`搜索所有以s开头的资源。

.. tip::
    
    如果想要搜索全部的Module，使用`hub search \*`并不生效，这是因为shell会自行进行通配符展开，将\*替换为当前目录下的文件名。为了进行全局搜索，用户可以直接键入`hub search`。


hub list
==================

列出本地已经安装的Module

hub run
==================

用于执行Module的预测，需要注意的是，并不是所有的模型都支持预测（同样，也不是所有的模型都支持迁移学习）。使用示例可以参考[hub run快速体验](../quick_experience/cmd_quick_run.md)。

PaddleHub尽量简化了用户在使用命令行预测时的理解成本，一般来讲，我们将预测分为NLP和CV两大类

NLP类的任务
---------------
输入数据通过--input_text指定。以百度LAC模型（中文词法分析）为例，可以通过以下命令实现文本分析。


.. code-block:: console

    $ hub run lac --input_text "今天是个好日子"


CV类的任务
---------------

输入数据通过`--input_path`指定。以SSD模型（单阶段目标检测）为例子，可以通过以下命令实现预测

.. code-block:: console

    $ hub run resnet_v2_50_imagenet --input_path test.jpg

hub help
==================

显示帮助信息

hub version
==================

显示PaddleHub版本信息

hub clear
==================

PaddleHub在使用过程中会产生一些缓存数据，这部分数据默认存放在${HUB_HOME}/.paddlehub/cache目录下，用户可以通过clear命令来清空缓存

hub config
==================

用于查看和设置paddlehub相关设置，包括对server地址、日志级别的设置：

.. code-block:: console

    $ # 显示当前paddlehub的设置
    $ hub config 
    
    $ # 恢复当前paddlehub的设置为默认设置
    $ hub config reset 
    
    $ # 设置当前paddlehub-server地址为${HOST}，paddlehub客户端从此地址获取模型信息
    $ hub config server==${HOST} 
    
    $ # 设置当前日志级别为${LEVEL}， 可选值为CRITICAL, ERROR, WARNING, EVAL, TRAIN, INFO, DEBUG, 从左到右优先级从高到低
    $ hub config log.level==${LEVEL} 
    
    $ # 设置当日志是否可用
    $ hub config log.enable==True|False 

hub serving
==================

用于一键部署Module预测服务，详细用法见`PaddleHub Serving一键服务部署 <serving>`_
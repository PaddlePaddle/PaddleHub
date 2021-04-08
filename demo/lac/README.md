# LAC 词法分析


本示例展示如何使用LAC Module进行预测。

LAC是中文词法分析模型，可以用于进行中文句子的分词/词性标注/命名实体识别等功能，关于模型的细节参见[模型介绍](https://www.paddlepaddle.org.cn/hubdetail?name=lac&en_category=LexicalAnalysis)。


## 命令行方式预测

`cli_demo.sh`给出了使用命令行接口(Command Line Interface)调用Module预测的示例脚本，
通过以下命令试验下效果。

```shell
$ hub run lac --input_text "今天是个好日子"
$ hub run lac --input_file test.txt --user_dict user.dict
```
test.txt 存放待分词文本， 如：

```text
今天是个好日子  
今天天气晴朗
```
user.dict为用户自定义词典，可以不指定，当指定自定义词典时，可以干预默认分词结果。
词典包含两列，第一列为单词，第二列为单词词性。以“/”分隔。词典样例如下：

```text
天气预报/n 
经/v
常/d
```

**NOTE:**

* 该PaddleHub Module使用词典干预功能时，依赖于第三方库pyahocorasick，请自行安装；
* 请不要直接复制示例文本使用，复制后的格式可能存在问题；


## 通过Python API预测

`lac_demo.py`给出了使用python API调用PaddleHub LAC Module预测的示例代码，
通过以下命令试验下效果。

```shell
python lac_demo.py
```

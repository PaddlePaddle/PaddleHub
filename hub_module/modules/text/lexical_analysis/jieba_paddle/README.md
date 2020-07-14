## 概述

该Module是jieba使用PaddlePaddle深度学习框架搭建的切词网络（双向GRU）。
同时也支持jieba的传统切词方法，如精确模式、全模式、搜索引擎模式等切词模式，使用方法和jieba保持一致。

更多信息参考：https://github.com/fxsjy/jieba

## API 说明

### cut(sentence, use_paddle=True, cut_all=False, HMM=True)

jieba_paddle预测接口，预测输入句子的分词结果

**参数**

* sentence(str): 单句预测数据。
* use_paddle(bool): 是否使用paddle模式（双向GRU）切词，默认为True。
* cut_all 参数用来控制是否采用全模式，默认为True；
* HMM 参数用来控制是否使用 HMM 模型， 默认为True；

**返回**

* results(list): 分词结果

### cut_for_search(sentence, HMM=True)

jieba的搜索引擎模式切词，该方法适合用于搜索引擎构建倒排索引的分词，粒度比较细

**参数**

* sentence(str): 单句预测数据。
* HMM 参数用来控制是否使用 HMM 模型， 默认为True；

### load_userdict(user_dict)

指定自己自定义的词典，以便包含 jieba 词库里没有的词。虽然 jieba 有新词识别能力，但是自行添加新词可以保证更高的正确率。

**参数**

* user_dict(str): 自定义词典的路径

### extract_tags(sentence, topK=20, withWeight=False, allowPOS=())

基于 TF-IDF 算法的关键词抽取

**参数**

* sentence(str): 待提取的文本
* topK(int): 返回几个 TF/IDF 权重最大的关键词，默认值为 20
* withWeight(bool): 为是否一并返回关键词权重值，默认值为 False
* allowPOS(tuple): 仅包括指定词性的词，默认值为空，即不筛选

### textrank(sentence, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))

基于 TextRank 算法的关键词抽取

**参数**

* sentence(str): 待提取的文本
* topK(int): 返回几个 TF/IDF 权重最大的关键词，默认值为 20
* withWeight(bool): 为是否一并返回关键词权重值，默认值为 False
* allowPOS(tuple): 仅包括指定词性的词，默认值为('ns', 'n', 'vn', 'v')

# Jieba_Paddle 服务部署

PaddleHub Serving可以部署一个切词服务，可以将此接口用于在线分词等web应用。

## 第一步：启动PaddleHub Serving

运行启动命令：
```shell
$ hub serving start -c serving_config.json
```

`serving_config.json`的内容如下：
```json
{
  "modules_info": {
    "jieba_paddle": {
      "init_args": {
        "version": "2.2.0"
      }
    }
  },
  "port": 8866,
  "use_singleprocess": false,
  "workers": 2
}
```

这样就完成了一个切词服务化API的部署，默认端口号为8866。


## 第二步：发送预测请求

配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

```python
import request
import json

# 待预测数据
text = "今天是个好日子"

# 设置运行配置
# 对应本地预测jieba_paddle.cut(sentence=text, use_paddle=True)
data = {"sentence": text, "use_paddle": True}

# 指定预测方法为jieba_paddle并发送post请求，content-type类型应指定json方式
# HOST_IP为服务器IP
url = "http://HOST_IP:8866/predict/jieba_paddle"
headers = {"Content-Type": "application/json"}
r = requests.post(url=url, headers=headers, data=json.dumps(data))

# 打印预测结果
print(json.dumps(r.json(), indent=4, ensure_ascii=False))
```

关于PaddleHub Serving更多信息参考[服务部署](https://github.com/PaddlePaddle/PaddleHub/blob/release/v1.6/docs/tutorial/serving.md)

## 查看代码

https://github.com/fxsjy/jieba

## 依赖

PaddlePaddle >= 1.8.0
PaddleHub >= 1.8.0

## 更新历史

* 1.0.0

    初始发布

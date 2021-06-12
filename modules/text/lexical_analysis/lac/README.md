## LAC API 说明

### \_\_init\_\_(user_dict=None)

构造LAC对象

**参数**

* user_dict(str): 自定义词典路径。如果需要使用自定义词典，则可通过该参数设置，否则不用传入该参数。

### cut(text, use_gpu=False, batch_size=1, return_tag=True)

lac预测接口，预测输入句子的分词结果

**参数**

* text(str or list): 待预测数据，单句预测数据（str类型）或者批量预测（list，每个元素为str
* use_gpu(bool): 是否使用GPU预测，如果使用GPU预测，则在预测之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置
* batch_size(int): 批处理大小
* return_tag(bool): 预测结果是否需要返回分词标签结果

### lexical_analysis(texts=[], data={}, use_gpu=False, batch_size=1, user_dict=None, return_tag=True)

**该接口将会在未来版本被废弃，如有需要，请使用cut接口预测**

lac预测接口，预测输入句子的分词结果

**参数**

* texts(list): 待预测数据，如果使用texts参数，则不用传入data参数，二选一即可
* data(dict): 预测数据，key必须为text，value是带预测数据。如果使用data参数，则不用传入texts参数，二选一即可。建议使用texts参数，data参数后续会废弃
* use_gpu(bool): 是否使用GPU预测，如果使用GPU预测，则在预测之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置
* batch_size(int): 批处理大小
* return_tag(bool): 预测结果是否需要返回分词标签结果

**返回**

* results(list): 分词结果

### context(trainable=False)

获取lac的预训练program以及program的输入输出变量

**参数**

* trainable(bool): trainable=True表示program中的参数在Fine-tune时需要微调，否则保持不变

**返回**

* inputs(dict): program的输入变量
* outputs(dict): program的输出变量
* main_program(Program): 带有预训练参数的program

### set_user_dict(dict_path)

加载用户自定义词典

**参数**

* dict_path(str ): 自定义词典路径

### del_user_dict()

删除自定义词典

### get_tags()

获取lac的标签

**返回**

* tag_name_dict(dict): lac的标签

## LAC 服务部署

PaddleHub Serving可以部署一个在线词法分析服务，可以将此接口用于词法分析、在线分词等在线web应用。

### 第一步：启动PaddleHub Serving

运行启动命令：
```shell
$ hub serving start -c serving_config.json
```

`serving_config.json`的内容如下：
```json
{
  "modules_info": {
    "lac": {
      "init_args": {
        "version": "2.1.0",
        "user_dict": "./test_dict.txt"
      },
      "predict_args": {}
    }
  },
  "port": 8866,
  "use_singleprocess": false,
  "workers": 2
}
```
其中user_dict含义为自定义词典路径，如果不使用lac自定义词典功能，则可以不填入。

这样就完成了一个词法分析服务化API的部署，默认端口号为8866。

**NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置。

### 第二步：发送预测请求

配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

```python
import requests
import json

# 待预测数据
text = ["今天是个好日子", "天气预报说今天要下雨"]

# 设置运行配置
# 对应本地预测lac.analysis_lexical(texts=text, batch_size=1, use_gpu=True)
data = {"texts": text, "batch_size": 1, "use_gpu":True}

# 指定预测方法为lac并发送post请求，content-type类型应指定json方式
# HOST_IP为服务器IP
url = "http://HOST_IP:8866/predict/lac"
headers = {"Content-Type": "application/json"}
r = requests.post(url=url, headers=headers, data=json.dumps(data))

# 打印预测结果
print(json.dumps(r.json(), indent=4, ensure_ascii=False))
```

关于PaddleHub Serving更多信息参考[服务部署](https://github.com/PaddlePaddle/PaddleHub/blob/release/v1.6/docs/tutorial/serving.md)

## 更新历史

* 1.0.0

    初始发布

* 1.0.1

  修复python2中编码问题

* 1.1.0

  支持词典干预，通过配置自定义词典可以对LAC默认分词结果进行干预

* 1.1.1

  修复输入文本中带有/字符时，使用词典干预会崩溃的问题

* 2.0.0

  修复输入文本为空、“ ”或者“\n”，使用LAC会崩溃的问题
  更新embedding_size, hidden_size为128，压缩模型，性能提升

* 2.1.0

  lac预测性能大幅提升
  支持是否返回分词标签tag，同时简化预测接口使用

* 2.1.1

  当输入文本为空字符串“”，返回切词后结果为空字符串“”，分词tag也为“”

* 2.2.0

  升级自定义词典功能，支持增加不属于lac默认提供的词性

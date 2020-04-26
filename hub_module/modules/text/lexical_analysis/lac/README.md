# LAC API 说明

## __init__(user_dict=None)

构造LAC对象

**参数**

* user_dict(str): 自定义词典路径。如果需要使用自定义词典，则可通过该参数设置，否则不用传入该参数。

## lexical_analysis(texts=[], data={}, use_gpu=False, batch_size=1, user_dict=None, return_tag=True)

lac预测接口，预测输入句子的分词结果

**参数**

* texts(list): 待预测数据，如果使用texts参数，则不用传入data参数，二选一即可
* data(dict): 预测数据，key必须为text，value是带预测数据。如果使用data参数，则不用传入texts参数，二选一即可。建议使用texts参数，data参数后续会废弃
* use_gpu(bool): 是否使用GPU预测，如果使用GPU预测，则在预测之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置
* batch_size(int): 批处理大小
* user_dict(None): 该参数不推荐使用，请在使用lexical_analysis()方法之前调用set_user_dict()方法设置自定义词典
* return_tag(bool): 预测结果是否需要返回分词标签结果

**返回**

* results(list): 分词结果

## context(trainable=False)

获取lac的预训练program以及program的输入输出变量

**参数**

* trainable(bool): trainable=True表示program中的参数在Fine-tune时需要微调，否则保持不变

**返回**

* inputs(dict): program的输入变量
* outputs(dict): program的输出变量
* main_program(Program): 带有预训练参数的program

## set_user_dict(dict_path)

加载用户自定义词典

**参数**

* dict_path(str ): 自定义词典路径

## del_user_dict()

删除自定义词典

## get_tags()

获取lac的标签

**返回**

* tag_name_dict(dict): lac的标签

# LAC 服务部署

PaddleHub Serving可以部署一个在线词法分析服务，可以将此接口用于词法分析、在线分词等在线web应用。

## 第一步：启动PaddleHub Serving

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
        "version": "2.1.0"
        "user_dict": "./test_dict.txt"
      }
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

## 第二步：发送预测请求

配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

```python
import request
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

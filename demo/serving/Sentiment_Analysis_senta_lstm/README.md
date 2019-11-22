## 数据格式  
input: {"text": [text_1, text_2, ...]}  
output: {"results":[result_1, result_2, ...]}  

## Serving快速启动命令  
```shell
$ hub serving start -m senta_lstm  
```

## python脚本
``` shell
$ python senta_lstm_serving_demo.py  
```  

## 结果示例  
```python
{  
    "results": [  
        {  
            "negative_probs": 0.7079,  
            "positive_probs": 0.2921,  
            "sentiment_key": "negative",  
            "sentiment_label": 0,  
            "text": "我不爱吃甜食"  
        },  
        {  
            "negative_probs": 0.0149,  
            "positive_probs": 0.9851,  
            "sentiment_key": "positive",  
            "sentiment_label": 1,  
            "text": "我喜欢躺在床上看电影"  
        }  
    ]  
}  
```

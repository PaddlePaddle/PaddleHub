## 数据格式  
input: {"text1": [text_a1, text_a2, ...], "text2": [text_b1, text_b2, ...]}  
output: {"results":[result_1, result_2, ...]}  

## Serving快速启动命令  
```shell
$ hub serving start -m simnet_bow  
```

## python脚本  
```shell
$ python simnet_bow_serving_demo.py  
```

## 结果示例  
```python
{  
    "results": [  
        {  
            "similarity": 0.8445,  
            "text_1": "这道题太难了",  
            "text_2": "这道题是上一年的考题"  
        },  
        {  
            "similarity": 0.9275,  
            "text_1": "这道题太难了",  
            "text_2": "这道题不简单"  
        },  
        {  
            "similarity": 0.9083,  
            "text_1": "这道题太难了",  
            "text_2": "这道题很有意思"  
        }  
    ]  
}  
```

## 数据格式  
input: {"text": [text_1, text_2, ...]}  
output: {"results":[result_1, result_2, ...]}  

## Serving快速启动命令  
```shell  
$ hub serving start -m lac  
```  

## python脚本  
#### 不携带用户自定义词典  
```shell  
$ python lac_no_dict_serving_demo.py  
```  

#### 携带用户自定义词典  
```shell  
$ python lac_with_dict_serving_demo.py  
```  

## 结果示例  
```python
{  
    "results": [  
        {  
            "tag": [  
                "TIME",  
                "v",  
                "q",  
                "n"  
            ],  
            "word": [  
                "今天",  
                "是",  
                "个",  
                "好日子"  
            ]  
        },  
        {  
            "tag": [  
                "n",  
                "v",  
                "TIME",  
                "v",  
                "v"  
            ],  
            "word": [  
                "天气预报",  
                "说",  
                "今天",  
                "要",  
                "下雨"  
            ]  
        }  
    ]  
}  
```

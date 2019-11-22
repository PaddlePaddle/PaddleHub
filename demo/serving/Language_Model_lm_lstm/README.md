## 数据格式  
input: {"text": [text_1, text_2, ...]}  
output: {"results":[result_1, result_2, ...]}  

## Serving快速启动命令  
```shell
$ hub serving start -m lm_lstm  
```

## python脚本  
```shell
$ python lm_lstm_serving_demo.py
```

## 结果示例
```python
{  
    "results": [  
        {  
            "perplexity": 4.584166451916099,  
            "text": "the plant which is owned by <unk> & <unk> co. was under            contract with <unk> to make the cigarette filter"  
        },  
        {  
            "perplexity": 6.038358983397484,  
            "text": "more common <unk> fibers are <unk> and are more easily            rejected by the body dr. <unk> explained"  
        }  
    ]  
}  
```  

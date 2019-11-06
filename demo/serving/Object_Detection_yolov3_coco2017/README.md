## 数据格式  
input: {files: {"image": [file_1, file_2, ...]}}  
output: {"results":[result_1, result_2, ...]}  

## Serving快速启动命令  
```shell  
$ hub serving start -m yolov3_coco2017  
```

## python脚本  
```shell
$ python yolov3_coco2017_serving_demo.py  
```

## 结果示例  
```python
[  
    {  
        "path": "cat.jpg",  
        "data": [  
            {  
                "left": 322.2323,  
                "right": 1420.4119,  
                "top": 208.81363,  
                "bottom": 996.04395,  
                "label": "cat",  
                "confidence": 0.9289875  
            }  
        ]  
    },  
    {  
        "path": "dog.jpg",  
        "data": [  
            {  
                "left": 204.74722,  
                "right": 746.02637,  
                "top": 122.793274,  
                "bottom": 566.6292,  
                "label": "dog",  
                "confidence": 0.86698055  
            }  
        ]  
    }  
]  
```
结果含有生成图片的base64编码，可提取生成图片，示例python脚本生成图片位置为当前目录下的output文件夹下。

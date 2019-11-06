## 数据格式  
input: {files: {"image": [file_1, file_2, ...]}}  
output: {"result":[result_1, result_2, ...]}  

## Serving快速启动命令  
$ hub serving start -m vgg11_imagenet  

## python脚本  
python vgg11_imagenet_serving_demo.py  

## 结果示例
```python
{  
    "results": "[[{'Egyptian cat': 0.540287435054779}], [{'daisy': 0.9976677298545837}]]"  
}  
```  

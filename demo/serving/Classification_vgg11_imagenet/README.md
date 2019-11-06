## 数据格式  
input: {files: {"image": [file_1, file_2, ...]}}  
output: {"results":[result_1, result_2, ...]}  

## Serving快速启动命令  
```shell
$ hub serving start -m vgg11_imagenet  
```  

## python脚本  
```shell
$ python vgg11_imagenet_serving_demo.py  
```  

## 结果示例
```python
{  
    "results": "[[{'Egyptian cat': 0.540287435054779}], [{'daisy': 0.9976677298545837}]]"  
}  
```  
结果含有生成图片的base64编码，可提取生成图片，示例python脚本生成图片位置为当前目录下的output文件夹下。

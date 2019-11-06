## 数据格式  
input: {files: {"image": [file_1, file_2, ...]}}  
output: {"result":[result_1, result_2, ...]}  

## Serving快速启动命令  
$ hub serving start -m deeplabv3p_xception65_humanseg  

## python脚本  
python deeplabv3p_xception65_humanseg_serving_demo.py


## 结果示例  
```python
[  
    {  
        "origin": "girl_2.jpg",  
        "processed": "humanseg_output/girl_2.png"  
    }  
]  
```
结果含有生成图片的base64编码，可提取生成图片，示例python脚本生成图片位置为当前目录下的output文件夹中。

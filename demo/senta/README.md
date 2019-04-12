## 关于Senta

https://github.com/baidu/Senta

## 创建Module
本目录包含了创建一个基于senta预训练模型的Module的脚本。
通过以下脚本来一键创建一个senta Module
```shell
$ sh create_module.sh
```
NOTE:
* 如果进行下面示例的脚本或者代码，请确保执行上述脚本
* 关于创建Module的API和细节，请查看`create_module.py`

## 使用Module预测
该Module创建完成后，可以通过命令行或者python API两种方式进行预测
### 命令行方式
`infer.sh`给出了使用命令行调用Module预测的示例脚本
通过以下命令试验下效果
```shell
$ sh infer.sh
```
### 通过python API
`infer_by_code.py`给出了使用python API调用Module预测的示例代码
通过以下命令试验下效果
```shell
$ python infer_by_code.py
```

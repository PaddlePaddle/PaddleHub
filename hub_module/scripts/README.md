# 打包&安装方法

以打包lac模型为例子，需要以下几个步骤:

## 一、配置文件

配置一个yml文件，格式如下:
> name: 模型名称
>
> dir: 模型目录（相对于repo根目录）
>
> exclude: 打包时需要跳过的文件名（相对于dir目录），可以不填写该字段
>
> resources: 需要额外下载的资源文件，可以不填写该字段
>
>> url: 资源所在的url
>>
>> dest: 资源下载后解压到什么路径（相对于dir目录）
>>
>> uncompress: 文件是否需要解压。可以不填写该字段，默认为False

## 二、执行打包代码

```shell
python pack.py --config configs/lac.yml
```
执行完该操作后，如果打包成功，则会保存一个lac-2.1.0.tar.gz文件

## 三、安装模型

```shell
hub install lac-2.1.0.tar.gz
```

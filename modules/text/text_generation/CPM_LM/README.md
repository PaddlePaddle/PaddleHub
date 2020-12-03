## 概述
CPM-LM 是一个基于 GPT-2 的预训练生成模型，参数规模达 26 亿，预训练中文数据规模 100 GB，使用了 64 块 V100 GPU，训练时间约为 3 周。能够在多种自然语言处理任务上进行零次学习或少次学习，并达到较好的效果。基于给定上文，模型可以续写出一致性高、可读性强的文本，达到现有中文生成模型的领先效果。

模型参数转换至官方开源项目，由于模型较大，推荐在GPU环境下运行，并且请确保运行环境的内存大于20G且显卡显存大于12G，否则可能无法正常运行

更多详情参考[清源CPM官网](https://cpm.baai.ac.cn)及其[Github项目主页](https://github.com/TsinghuaAI/CPM-Generate)

## API
```python
def predict(text, max_len=32, end_word=None):
```
预测 API ，根据输入的文字进行文本生成，使用 Greedy Search 进行解码。

**参数**
* text (str) : 输入文本
* max_len (int) : 生成文本的最大长度
* end_word (str or None) : 终止生成的标志词

**返回**
* results (str): 生成的文本

```python
def tokenizer.encode(text):
```
编码 API

**参数**
* text (str) : 输入文本

**返回**
* results (list[int]) : 输出编码

```python
def tokenizer.decode(ids):
```
解码 API

**参数**
* ids (list[int]) : 输入编码

**返回**
* results (str) : 输出文本

```python
def model(x, kv_cache=None, use_cache=False):
```
模型前向计算 API

**参数**
* x (tensor) : 输入编码
* kv_cache (tensor) : 输入的缓存
* use_cache (bool) : 是否使用缓存

**返回**
* results (tensor) : 模型输出

**代码示例**
```python
import paddlehub as hub

model = hub.Module(name='CPM_LM')
```
```python
inputs = '''默写古诗:
日照香炉生紫烟，遥看瀑布挂前川。
飞流直下三千尺，'''
outputs = model.predict(inputs, max_len=10, end_word='\n')
print(inputs+outputs)
```
> 默写古诗:  
日照香炉生紫烟，遥看瀑布挂前川。  
飞流直下三千尺，疑是银河落九天。
```python
inputs = '''问题：西游记是谁写的？
答案：'''
outputs = model.predict(inputs, max_len=10, end_word='\n')
print(inputs+outputs)
```
> 问题：西游记是谁写的？  
答案：吴承恩。
```python
inputs = '''小明决定去吃饭，小红继续写作业
问题：去吃饭的人是谁？
答案：'''
outputs = model.predict(inputs, max_len=10, end_word='\n')
print(inputs+outputs)
```
> 小明决定去吃饭，小红继续写作业  
问题：去吃饭的人是谁？  
答案：小明
```python
inputs = '''默写英文：
狗：dog
猫：'''
outputs = model.predict(inputs, max_len=10, end_word='\n')
print(inputs+outputs)
```
> 默写英文：  
狗：dog  
猫：cat

## 查看代码
https://github.com/jm12138/CPM-Generate-Paddle

## 依赖
paddlepaddle >= 2.0.0rc0  
paddlehub >= 2.0.0b1
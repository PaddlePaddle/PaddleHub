# ERNIE3_Zeus

|模型名称|ERNIE3_Zeus|
| :--- | :---: |
|类别|文本-文本生成|
|网络|-|
|数据集|-|
|是否支持Fine-tuning|否|
|模型大小|-|
|最新更新日期|2022-08-16|
|数据指标|-|

## 一、模型基本信息
### 应用效果展示
* 作文创作：
    * 作文标题：可为与有为 
    
    * 题目内容：当今社会竞争加剧，每人身上压力倍增，于是乎，“躺平”成了话语，“佛系”成了主义。仰望星空的眼神逐渐空洞，低头专注的仅仅是尺寸之间，这些“万事不可为”之信念确乎不该是我们青年一辈所应有所该有。请谨记，相信“可为”才能实现梦想，坚持“有为”才能书写华章。

    * 作文：当今社会竞争加剧，每人身上压力倍增，于是乎，“躺平”成为话语，“佛系”成为主义。仰望星空的眼神逐渐空洞，低头专注的仅仅是尺寸之间，这些“万事不可为”之信念确乎不该是我们青年一辈所应有所该有。请谨记，相信“可为”才能实现梦想，坚持“有为”才能书写华章。相信“可为”才能实现梦想——“可为”才能成就梦想。古往今来，人类的梦想从没有停止变换过。

* 文案创作：
    * 产品：护手霜

    * 文案：这款护手霜质地非常的轻薄水润，涂开后能很快滋润皮肤，不油腻，很滋润，还不会长脂肪颗粒，不会有粘黏感，用过后很干爽，还不会有黏腻感。

### 模型介绍
* ERNIE 3.0 Zeus 是 ERNIE 3.0 系列模型的最新升级。

* 其除了对无标注数据和知识图谱的学习之外，还通过持续学习对百余种不同形式的任务数据学习。

* 实现了任务知识增强，显著提升了模型的零样本/小样本学习能力。

* 更多详情参考 [文心大模型官网](https://wenxin.baidu.com/wenxin) 及 [ERNIE 3.0 Zeus 项目主页](https://wenxin.baidu.com/wenxin/modelbasedetail/ernie3_zeus)。

## 二、安装
### 1、环境依赖
* paddlepaddle >= 2.0.0

* paddlehub >= 2.2.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

### 2、安装
  - ```shell
    $ hub install ERNIE3_Zeus
    ```

* 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

### 3. 使用申请
* 请前往 [文心旸谷社区](https://wenxin.baidu.com/moduleApi/key) 申请使用本模型所需的 API key 和 Secret Key。


## 三、模型 API 预测
### 1. 命令行预测

- ```bash
  # 作文创作
  $ hub run ERNIE3_Zeus \
        --task article_creation \
        --api_key [api_key] \
        --secret_key [secret_key] \
        --text '可为与有为' '当今社会竞争加剧，每人身上压力倍增，于是乎，“躺平”成了话语，“佛系”成了主义。仰望星空的眼神逐渐空洞，低头专注的仅仅是尺寸之间，这些“万事不可为”之信念确乎不该是我们青年一辈所应有所该有。请谨记，相信“可为”才能实现梦想，坚持“有为”才能书写华章。'
    ```

        当今社会竞争加剧，每人身上压力倍增，于是乎，“躺平”成为话语，“佛系”成为主义。仰望星空的眼神逐渐空洞，低头专注的仅仅是尺寸之间，这些“万事不可为”之信念确乎不该是我们青年一辈所应有所该有。请谨记，相信“可为”才能实现梦想，坚持“有为”才能书写华章。相信“可为”才能实现梦想——“可为”才能成就梦想。古往今来，人类的梦想从没有停止变换过。

- **参数**
    * --task(str): API 名称如 “copywriting_creation”、“article_creation” 等。
    * --text(List[str]): 输入文本可根据 API 需要速入一段或两段文本，以空格做切分。
    * 其他参数与自定义文本生成 API 相同。

### 2. 预测代码示例
- ```python
    import paddlehub as hub

    # 加载模型
    ERNIE3_Zeus = hub.Module(name='ERNIE3_Zeus')

    # 设置 api_key / secret_key
    api_key = [api_key]
    secret_key = [secret_key]
    
    # 作文创作
    title = '可为与有为'
    context = '当今社会竞争加剧，每人身上压力倍增，于是乎，“躺平”成了话语，“佛系”成了主义。仰望星空的眼神逐渐空洞，低头专注的仅仅是尺寸之间，这些“万事不可为”之信念确乎不该是我们青年一辈所应有所该有。请谨记，相信“可为”才能实现梦想，坚持“有为”才能书写华章。'
    article = ERNIE3_Zeus.article_creation(
        api_key=api_key,
        secret_key=secret_key,
        text=[title, context]
    )

    print(article)
    ```
    当今社会竞争加剧，每人身上压力倍增，于是乎，“躺平”成为话语，“佛系”成为主义。仰望星空的眼神逐渐空洞，低头专注的仅仅是尺寸之间，这些“万事不可为”之信念确乎不该是我们青年一辈所应有所该有。请谨记，相信“可为”才能实现梦想，坚持“有为”才能书写华章。相信“可为”才能实现梦想——“可为”才能成就梦想。古往今来，人类的梦想从没有停止变换过。


### 3. API
```python
def custom_generation(
    api_key: str, 
    secret_key: str, 
    text: str, 
    seq_len: int = 256, 
    task_prompt: str = '', 
    dataset_prompt: str = '', 
    topk: int = 10,
    temperature: float = 1.0, 
    penalty_score: float = 1.0, 
    penalty_text: str = '',  
    choice_text: str = '', 
    stop_token: str = '',
    is_unidirectional: bool = False, 
    min_dec_len: int = 1, 
    min_dec_penalty_text: str = ''
) -> str
```
自定义文本生成 API

**参数**
* api_key(str): API Key。
* secret_key(str): Secret Key。
* text(srt): 输入内容, 长度不超过 1000。
* seq_len(int): 输出内容最大长度, 长度不超过 1000。
* task_prompt(str): 任务类型的模板。
* dataset_prompt(str): 数据集类型的模板。
* topk(int): topk采样, 取值 > 1, 默认为 10。每步的生成的结果从 topk 的概率值分布中采样。其中 topk = 1 表示贪婪采样, 每次生成结果固定。
* temperature(float): 温度系数, 取值 > 0.0, 默认为 1.0。更大的温度系数表示模型生成的多样性更强。
* penalty_score(float): 重复惩罚。取值 >= 1.0, 默认为 1.0。通过对已生成的 token 增加惩罚, 减少重复生成的现象。值越大表示惩罚越大。
* penalty_text(str): 惩罚文本, 默认为空。模型无法生成该字符串中的 token。通过设置该值, 可以减少某些冗余与异常字符的生成。
* choice_text(str): 候选文本, 默认为空。模型只能生成该字符串中的 token 的组合。通过设置该值, 可以对某些抽取式任务进行定向调优。
* stop_token(str): 提前结束符, 默认为空。预测结果解析时使用的结束字符, 碰到对应字符则直接截断并返回。可以通过设置该值, 过滤掉 few-shot 等场景下模型重复的 cases。
* is_unidirectional(bool): 单双向控制开关, 取值 0 或者 1, 默认为 0。0 表示模型为双向生成, 1 表示模型为单向生成。建议续写与 few-shot 等通用场景建议采用单向生成方式, 而完型填空等任务相关场景建议采用双向生成方式。
* min_dec_len(int): 最小生成长度, 取值 >= 1, 默认为 1。开启后会屏蔽掉 END 结束符号, 让模型生成至指定的最小长度。
* min_dec_penalty_text(str): 默认为空, 与最小生成长度搭配使用, 可以在 min_dec_len 步前不让模型生成该字符串中的 tokens。

**返回**
* text(str): 生成的文本。

```python
def article_creation(
    api_key: str, 
    secret_key: str, 
    text: List[str], 
    dataset_prompt: str = 'zuowen', 
    **kwargs
) -> str
```
作文创作 API

**参数**
* text(List[str]): [作文标题, 作文题目内容]。
* dataset_prompt(str): 数据集类型的模板，作文数据集模板为：“zuowen”。
* 其他参数与自定义文本生成 API 相同。

**返回**
* text(str): 生成的作文。

```python
def copywriting_creation(
    api_key: str, 
    secret_key: str, 
    text: str, 
    **kwargs
) -> str
```
文案创作 API

**参数**
* text(str): 商品文字。
* 其他参数与自定义文本生成 API 相同。

**返回**
* text(str): 生成的文案。

```python
def text_summarization(
    api_key: str, 
    secret_key: str, 
    text: str, 
    task_prompt: str = 'Summarization', 
    **kwargs
) -> str
```
摘要生成 API

**参数**
* text(str): 段落文本。
* task_prompt(str): 任务类型的模板，摘要生成任务模版为：“Summarization”。
* 其他参数与自定义文本生成 API 相同。

**返回**
* text(str): 生成的摘要。


```python
def question_generation(
    api_key: str, 
    secret_key: str, 
    text: str, 
    task_prompt: str = 'QuestionGeneration', 
    **kwargs
) -> str
```
问题生成 API

**参数**
* text(str): 提问资料。
* task_prompt(str): 任务类型的模板，问题生成任务模版为：“QuestionGeneration”。
* 其他参数与自定义文本生成 API 相同。

**返回**
* text(str): 生成的问题。


```python
def poetry_creation(
    api_key: str, 
    secret_key: str, 
    text: List[str], 
    dataset_prompt: str = 'poetry', 
    **kwargs
) -> str
```
古诗创作 API

**参数**
* text(List[str]): [古诗标题, 古诗内容]。
* dataset_prompt(str): 数据集类型的模板，诗歌数据集模板为：“poetry”。
* 其他参数与自定义文本生成 API 相同。

**返回**
* text(str): 生成的古诗。

```python
def couplet_continuation(
    api_key: str, 
    secret_key: str, 
    text: str, 
    dataset_prompt: str = 'couplet', 
    **kwargs
) -> str
```
对联续写 API

**参数**
* text(str): 上联文本。
* dataset_prompt(str): 数据集类型的模板，对联数据集模板为：“couplet”。
* 其他参数与自定义文本生成 API 相同。

**返回**
* text(str): 生成的下联。

```python
def answer_generation(
    api_key: str, 
    secret_key: str, 
    text: str, 
    **kwargs
) -> str
```
自由问答 API

**参数**
* text(str): 问题文本。
* 其他参数与自定义文本生成 API 相同。

**返回**
* text(str): 生成的回答。

```python
def article_continuation(
    api_key: str, 
    secret_key: str, 
    text: str, 
    **kwargs
) -> str
```
小说续写 API

**参数**
* text(str): 小说上文。
* 其他参数与自定义文本生成 API 相同。

**返回**
* text(str): 生成的下文。

```python
def sentiment_classification(
    api_key: str, 
    secret_key: str, 
    text: str, 
    task_prompt: str = 'SentimentClassification',
    **kwargs
) -> str
```
情感分析 API

**参数**
* text(str): 情感分析文本。
* task_prompt(str): 任务类型的模板，情感分析任务模版为：“Summarization”。
* 其他参数与自定义文本生成 API 相同。

**返回**
* text(str): 分析结果。

```python
def information_extraction(
    api_key: str, 
    secret_key: str, 
    text: List[str], 
    task_prompt: str = 'QA_MRC', 
    **kwargs
) -> str
```
信息抽取 API

**参数**
* text(List[str]): [问题材料, 问题]。
* task_prompt(str): 任务类型的模板，信息抽取任务模版为：“QA_MRC”。
* 其他参数与自定义文本生成 API 相同。

**返回**
* text(str): 抽取到的信息。

```python
def synonymous_rewriting(
    api_key: str, 
    secret_key: str, 
    text: str, 
    task_prompt: str = 'Paraphrasing',
    **kwargs
) -> str
```
同义改写 API

**参数**
* text(str): 待改写文本。
* task_prompt(str): 任务类型的模板，同义改写任务模版为：“Paraphrasing”。
* 其他参数与自定义文本生成 API 相同。

**返回**
* text(str): 改写后的文本。

```python
def semantic_matching(
    api_key: str, 
    secret_key: str, 
    text: List[str], 
    task_prompt: str = 'SemanticMatching', 
    **kwargs
) -> str
```
文本匹配 API

**参数**
* text(List[str]): [文本1, 文本2]。
* task_prompt(str): 任务类型的模板，同义改写任务模版为：“SemanticMatching”。
* 其他参数与自定义文本生成 API 相同。

**返回**
* text(str): 匹配结果。


```python
def text_correction(
    api_key: str, 
    secret_key: str, 
    text: str, 
    task_prompt: str = 'Correction', 
    **kwargs
) -> str
```
文本纠错 API

**参数**
* text(str): 待纠错文本。
* task_prompt(str): 任务类型的模板，同义改写任务模版为：“Correction”。
* 其他参数与自定义文本生成 API 相同。

**返回**
* text(str): 纠错后文本。

```python
def text_cloze(
    api_key: str, 
    secret_key: str, 
    text: str, 
    **kwargs
) -> str
```
完形填空 API

**参数**
* text(str): 待填空文本，填空处使用 [MASK] 标记占位。
* 其他参数与自定义文本生成 API 相同。

**返回**
* text(str): 填空后文本。


```python
def text2SQL(
    api_key: str, 
    secret_key: str, 
    text: str, 
    task_prompt: str = 'Text2SQL', 
    **kwargs
) -> str
```
Text2SQL API

**参数**
* text(str): SQL 语句文本。
* task_prompt(str): 任务类型的模板，Text2SQL 任务模版为：“Text2SQL”。
* 其他参数与自定义文本生成 API 相同。

**返回**
* text(str): 生成的 SQL 语句。

## 四、更新历史
* 1.0.0 

  初始发布

  ```shell
  $ hub install ERNIE3_Zeus == 1.0.0
  ```
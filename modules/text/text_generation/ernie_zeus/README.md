# ernie_zeus

|模型名称|ernie_zeus|
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
- 作文创作：
    - 作文标题：诚以养德，信以修身

    - 作文：翻开我的书橱，展现在眼前的就是《曾国藩家书》。每当读起这些充满哲理的内容时，心里总会不禁佩服他。他虽出生于官宦之家，但并没有因此而骄傲自大，从小养成了平淡做人、踏实肯干的好品质，最后更赢得了属下和朋友们对他的一致认同和赞赏。由此可见，只要平时注意锻炼自己，处事脚踏实地，定能收获一番丰硕的成果！记得有句话叫“以诚待人”。我觉得曾国藩就是始终把做到真诚与诚信作为修身立业的准则和美德。

- 文案创作：
    - 产品描述：芍药香氛的沐浴乳

    - 文案：使用多种纯天然草本植物精华，泡沫细腻绵密，丰富的维他命及矿物质滋养皮肤。成分温和安全，适合干性、中性肌肤或敏感性肌肤使用！

### 模型介绍
ERNIE 3.0 Zeus 是 ERNIE 3.0 系列模型的最新升级。其除了对无标注数据和知识图谱的学习之外，还通过持续学习对百余种不同形式的任务数据学习。实现了任务知识增强，显著提升了模型的零样本/小样本学习能力。

更多详情参考 [文心大模型官网](https://wenxin.baidu.com/wenxin) 及 [ERNIE 3.0 Zeus 项目主页](https://wenxin.baidu.com/wenxin/modelbasedetail/ernie3_zeus)。

## 二、安装
- ### 1、环境依赖

  - paddlepaddle >= 2.0.0

  - paddlehub >= 2.2.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install ernie_zeus
    ```

  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

- ### 3. 使用申请（可选）
  - 请前往 [文心旸谷社区](https://wenxin.baidu.com/moduleApi/key) 申请使用本模型所需的 API key 和 Secret Key。


## 三、模型 API 预测
- ### 1. 命令行预测

  - ```bash
    # 作文创作
    $ hub run ernie_zeus \
        --task composition_generation \
        --text '诚以养德，信以修身' 
    ```

    - **参数**
      - --task(str): 指定任务名称，与 API 名称保持一直
      - --text(str): 根据不同的任务输入所需的文本。
      - 其他参数请参考后续 API 章节。

- ### 2. 预测代码示例

  - ```python
    import paddlehub as hub

    # 加载模型
    model = hub.Module(name='ernie_zeus')

    # 作文创作
    result = model.composition_generation(
        text='诚以养德，信以修身'
    )

    print(result)
    ```

- ### 3. API
  - ```python
    def __init__(
        api_key: str = '', 
        secret_key: str = ''
    ) -> None
    ```
    
    - 初始化 API

    - **参数**

      - api_key(str): API Key。（可选）
      - secret_key(str): Secret Key。（可选）

  - ```python
    def custom_generation(
        text: str,
        min_dec_len: int = 1,
        seq_len: int = 128,
        topp: float = 1.0,
        penalty_score: float = 1.0,
        stop_token: str = '',
        task_prompt: str = '',
        penalty_text: str = '',
        choice_text: str = '',
        is_unidirectional: bool = False,
        min_dec_penalty_text: str = '',
        logits_bias: int = -10000,
        mask_type: str = 'word',
        api_key: str = '',
        secret_key: str = ''
    ) -> str
    ```
    - 自定义文本生成 API

    - **参数**
      - text(srt): 模型的输入文本, 为 prompt 形式的输入。文本长度 [1, 1000]。注: ERNIE 3.0-1.5B 模型取值范围 ≤ 512。
      - min_dec_len(int): 输出结果的最小长度, 避免因模型生成 END 或者遇到用户指定的 stop_token 而生成长度过短的情况,与 seq_len 结合使用来设置生成文本的长度范围 [1, seq_len]。
      - seq_len(int): 输出结果的最大长度, 因模型生成 END 或者遇到用户指定的 stop_token, 实际返回结果可能会小于这个长度, 与 min_dec_len 结合使用来控制生成文本的长度范围 [1, 1000]。(注: ERNIE 3.0-1.5B 模型取值范围 ≤ 512)
      - topp(float): 影响输出文本的多样性, 取值越大, 生成文本的多样性越强。取值范围 [0.0, 1.0]。
      - penalty_score(float): 通过对已生成的 token 增加惩罚, 减少重复生成的现象。值越大表示惩罚越大。取值范围 [1.0, 2.0]。
      - stop_token(str): 预测结果解析时使用的结束字符串, 碰到对应字符串则直接截断并返回。可以通过设置该值, 过滤掉 few-shot 等场景下模型重复的 cases。
      - task_prompt(str): 指定预置的任务模板, 效果更好。
        PARAGRAPH: 引导模型生成一段文章; SENT: 引导模型生成一句话; ENTITY: 引导模型生成词组; 
        Summarization: 摘要; MT: 翻译; Text2Annotation: 抽取; Correction: 纠错; 
        QA_MRC: 阅读理解; Dialogue: 对话; QA_Closed_book: 闭卷问答; QA_Multi_Choice: 多选问答; 
        QuestionGeneration: 问题生成; Paraphrasing: 复述; NLI: 文本蕴含识别; SemanticMatching: 匹配; 
        Text2SQL: 文本描述转SQL; TextClassification: 文本分类; SentimentClassification: 情感分析; 
        zuowen: 写作文; adtext: 写文案; couplet: 对对联; novel: 写小说; cloze: 文本补全; Misc: 其它任务。
      - penalty_text(str): 模型会惩罚该字符串中的 token。通过设置该值, 可以减少某些冗余与异常字符的生成。
      - choice_text(str): 模型只能生成该字符串中的 token 的组合。通过设置该值, 可以对某些抽取式任务进行定向调优。
      - is_unidirectional(bool): False 表示模型为双向生成, True 表示模型为单向生成。建议续写与 few-shot 等通用场景建议采用单向生成方式, 而完型填空等任务相关场景建议采用双向生成方式。
      - min_dec_penalty_text(str): 与最小生成长度搭配使用, 可以在 min_dec_len 步前不让模型生成该字符串中的 tokens。
      - logits_bias(int): 配合 penalty_text 使用, 对给定的 penalty_text 中的 token 增加一个 logits_bias, 可以通过设置该值屏蔽某些 token 生成的概率。
      - mask_type(str): 设置该值可以控制模型生成粒度。可选参数为 word, sentence, paragraph。

    - **返回**
      - text(str): 生成的文本。

  - ```python
     def text_cloze(
         text: str,
         min_dec_len: int = 1,
         seq_len: int = 512,
         topp: float = 0.9,
         penalty_score: float = 1.0
     ) -> str
     ```

     - 完形填空 API

     - **参数**
       - text(str): 文字段落。使用 [MASK] 标记待补全文字。
       - min_dec_len(int): 输出结果的最小长度, 避免因模型生成 END 或者遇到用户指定的 stop_token 而生成长度过短的情况,与 seq_len 结合使用来设置生成文本的长度范围 [1, seq_len]。
       - seq_len(int): 输出结果的最大长度, 因模型生成 END 或者遇到用户指定的 stop_token, 实际返回结果可能会小于这个长度, 与 min_dec_len 结合使用来控制生成文本的长度范围 [1, 1000]。(注: ERNIE 3.0-1.5B 模型取值范围 ≤ 512)
       - topp(float): 影响输出文本的多样性, 取值越大, 生成文本的多样性越强。取值范围 [0.0, 1.0]。
       - penalty_score(float): 通过对已生成的 token 增加惩罚, 减少重复生成的现象。值越大表示惩罚越大。取值范围 [1.0, 2.0]。

     - **返回**
       - text(str): 补全词语

  - ```python
     def composition_generation(
         text: str,
         min_dec_len: int = 128,
         seq_len: int = 512,
         topp: float = 0.9,
         penalty_score: float = 1.2
     ) -> str
     ```
     - 作文创作 API

     - **参数**
       - text(str): 作文题目。
       - min_dec_len(int): 输出结果的最小长度, 避免因模型生成 END 或者遇到用户指定的 stop_token 而生成长度过短的情况,与 seq_len 结合使用来设置生成文本的长度范围 [1, seq_len]。
       - seq_len(int): 输出结果的最大长度, 因模型生成 END 或者遇到用户指定的 stop_token, 实际返回结果可能会小于这个长度, 与 min_dec_len 结合使用来控制生成文本的长度范围 [1, 1000]。(注: ERNIE 3.0-1.5B 模型取值范围 ≤ 512)
       - topp(float): 影响输出文本的多样性, 取值越大, 生成文本的多样性越强。取值范围 [0.0, 1.0]。
       - penalty_score(float): 通过对已生成的 token 增加惩罚, 减少重复生成的现象。值越大表示惩罚越大。取值范围 [1.0, 2.0]。

     - **返回**
       - text(str): 作文内容。

  - ```python
     def answer_generation(
         text: str,
         min_dec_len: int = 2,
         seq_len: int = 512,
         topp: float = 0.9,
         penalty_score: float = 1.2
     ) -> str
     ```
     - 自由问答 API

     - **参数**
       - text(str): 问题内容。
       - min_dec_len(int): 输出结果的最小长度, 避免因模型生成 END 或者遇到用户指定的 stop_token 而生成长度过短的情况,与 seq_len 结合使用来设置生成文本的长度范围 [1, seq_len]。
       - seq_len(int): 输出结果的最大长度, 因模型生成 END 或者遇到用户指定的 stop_token, 实际返回结果可能会小于这个长度, 与 min_dec_len 结合使用来控制生成文本的长度范围 [1, 1000]。(注: ERNIE 3.0-1.5B 模型取值范围 ≤ 512)
       - topp(float): 影响输出文本的多样性, 取值越大, 生成文本的多样性越强。取值范围 [0.0, 1.0]。
       - penalty_score(float): 通过对已生成的 token 增加惩罚, 减少重复生成的现象。值越大表示惩罚越大。取值范围 [1.0, 2.0]。

     - **返回**
       - text(str): 问题答案。


   - ```python
     def couplet_continuation(
         text: str,
         min_dec_len: int = 2,
         seq_len: int = 512,
         topp: float = 0.9,
         penalty_score: float = 1.0
     ) -> str
     ```
     - 对联续写 API

     - **参数**
       - text(str): 对联上联。
       - min_dec_len(int): 输出结果的最小长度, 避免因模型生成 END 或者遇到用户指定的 stop_token 而生成长度过短的情况,与 seq_len 结合使用来设置生成文本的长度范围 [1, seq_len]。
       - seq_len(int): 输出结果的最大长度, 因模型生成 END 或者遇到用户指定的 stop_token, 实际返回结果可能会小于这个长度, 与 min_dec_len 结合使用来控制生成文本的长度范围 [1, 1000]。(注: ERNIE 3.0-1.5B 模型取值范围 ≤ 512)
       - topp(float): 影响输出文本的多样性, 取值越大, 生成文本的多样性越强。取值范围 [0.0, 1.0]。
       - penalty_score(float): 通过对已生成的 token 增加惩罚, 减少重复生成的现象。值越大表示惩罚越大。取值范围 [1.0, 2.0]。

     - **返回**
       - text(str): 对联下联。

   - ```python
     def copywriting_generation(
         text: str,
         min_dec_len: int = 32,
         seq_len: int = 512,
         topp: float = 0.9,
         penalty_score: float = 1.2
     ) -> str
     ```
     - 文案创作 API

     - **参数**
       - text(str): 产品描述。
       - min_dec_len(int): 输出结果的最小长度, 避免因模型生成 END 或者遇到用户指定的 stop_token 而生成长度过短的情况,与 seq_len 结合使用来设置生成文本的长度范围 [1, seq_len]。
       - seq_len(int): 输出结果的最大长度, 因模型生成 END 或者遇到用户指定的 stop_token, 实际返回结果可能会小于这个长度, 与 min_dec_len 结合使用来控制生成文本的长度范围 [1, 1000]。(注: ERNIE 3.0-1.5B 模型取值范围 ≤ 512)
       - topp(float): 影响输出文本的多样性, 取值越大, 生成文本的多样性越强。取值范围 [0.0, 1.0]。
       - penalty_score(float): 通过对已生成的 token 增加惩罚, 减少重复生成的现象。值越大表示惩罚越大。取值范围 [1.0, 2.0]。

     - **返回**
       - text(str): 产品文案。

  - ```python
     def novel_continuation(
         text: str,
         min_dec_len: int = 2,
         seq_len: int = 512,
         topp: float = 0.9,
         penalty_score: float = 1.2
     ) -> str
     ```
     - 小说续写 API

     - **参数**
       - text(str): 小说上文。
       - min_dec_len(int): 输出结果的最小长度, 避免因模型生成 END 或者遇到用户指定的 stop_token 而生成长度过短的情况,与 seq_len 结合使用来设置生成文本的长度范围 [1, seq_len]。
       - seq_len(int): 输出结果的最大长度, 因模型生成 END 或者遇到用户指定的 stop_token, 实际返回结果可能会小于这个长度, 与 min_dec_len 结合使用来控制生成文本的长度范围 [1, 1000]。(注: ERNIE 3.0-1.5B 模型取值范围 ≤ 512)
       - topp(float): 影响输出文本的多样性, 取值越大, 生成文本的多样性越强。取值范围 [0.0, 1.0]。
       - penalty_score(float): 通过对已生成的 token 增加惩罚, 减少重复生成的现象。值越大表示惩罚越大。取值范围 [1.0, 2.0]。

     - **返回**
       - text(str): 小说下文。

  - ```python
     def text_summarization(
         text: str,
         min_dec_len: int = 4,
         seq_len: int = 512,
         topp: float = 0.0,
         penalty_score: float = 1.0
     ) -> str
     ```
     - 文本摘要 API

     - **参数**
       - text(str): 文本段落。
       - min_dec_len(int): 输出结果的最小长度, 避免因模型生成 END 或者遇到用户指定的 stop_token 而生成长度过短的情况,与 seq_len 结合使用来设置生成文本的长度范围 [1, seq_len]。
       - seq_len(int): 输出结果的最大长度, 因模型生成 END 或者遇到用户指定的 stop_token, 实际返回结果可能会小于这个长度, 与 min_dec_len 结合使用来控制生成文本的长度范围 [1, 1000]。(注: ERNIE 3.0-1.5B 模型取值范围 ≤ 512)
       - topp(float): 影响输出文本的多样性, 取值越大, 生成文本的多样性越强。取值范围 [0.0, 1.0]。
       - penalty_score(float): 通过对已生成的 token 增加惩罚, 减少重复生成的现象。值越大表示惩罚越大。取值范围 [1.0, 2.0]。

     - **返回**
       - text(str): 段落摘要。
## 四、更新历史

* 1.0.0 

  初始发布

  ```shell
  $ hub install ernie_zeus == 1.0.0
  ```
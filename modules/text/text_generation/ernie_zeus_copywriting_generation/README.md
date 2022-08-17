# ernie_zeus_copywriting_generation

|模型名称|ernie_zeus_copywriting_generation|
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
    $ hub install ernie_zeus_copywriting_generation
    ```

  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

- ### 3. 使用申请（可选）
  - 请前往 [文心旸谷社区](https://wenxin.baidu.com/moduleApi/key) 申请使用本模型所需的 API key 和 Secret Key。


## 三、模型 API 预测
- ### 1. 命令行预测

  - ```bash
    $ hub run ernie_zeus_copywriting_generation \
            --text '芍药香氛的沐浴乳' 
    ```

    - **参数**
      - --text(str): 产品描述。
      - --min_dec_len(int): 输出结果的最小长度, 避免因模型生成 END 或者遇到用户指定的 stop_token 而生成长度过短的情况,与 seq_len 结合使用来设置生成文本的长度范围 [1, seq_len]。
      - --seq_len(int): 输出结果的最大长度, 因模型生成 END 或者遇到用户指定的 stop_token, 实际返回结果可能会小于这个长度, 与 min_dec_len 结合使用来控制生成文本的长度范围 [1, 1000]。(注: ERNIE 3.0-1.5B 模型取值范围 ≤ 512)
      - --topp(float): 影响输出文本的多样性, 取值越大, 生成文本的多样性越强。取值范围 [0.0, 1.0]。
      - --penalty_score(float): 通过对已生成的 token 增加惩罚, 减少重复生成的现象。值越大表示惩罚越大。取值范围 [1.0, 2.0]。

- ### 2. 预测代码示例
  - ```python
    import paddlehub as hub

    # 加载模型
    model = hub.Module(name='ernie_zeus_copywriting_generation')

    # 文案创作
    result = model.copywriting_generation(
        text='芍药香氛的沐浴乳' 
    )

    print(result)
    ```

- ### 3. API
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

## 四、更新历史
* 1.0.0 

  初始发布

  ```shell
  $ hub install ernie_zeus_copywriting_generation == 1.0.0
  ```
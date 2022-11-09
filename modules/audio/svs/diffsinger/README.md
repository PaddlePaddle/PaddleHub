# diffsinger

|模型名称|diffsinger|
| :--- | :---: |
|类别|音频-歌声合成|
|网络|DiffSinger|
|数据集|-|
|是否支持Fine-tuning|否|
|模型大小|256.1MB|
|指标|-|
|最新更新日期|2022-10-25|


## 一、模型基本信息

- ### 应用效果展示

  - 网络结构：
      <p align="center">
        <img src="https://neuralsvb.github.io/resources/model_all7.png"/>
      </p>

  - 样例结果示例：

    |文本|音频|
    |:-:|:-:|
    |让 梦 恒 久 比 天 长|<audio controls="controls"><source src="https://diffsinger.github.io/audio/singing_demo/diffsinger-base/000000007.wav" autoplay=""></audio>|
    |我 终 于 翱 翔|<audio controls="controls"><source src="https://diffsinger.github.io/audio/singing_demo/diffsinger-base/000000005.wav" autoplay=""></audio>|

- ### 模型介绍

  - DiffSinger，一个基于扩散概率模型的 SVS 声学模型。DiffSinger 是一个参数化的马尔科夫链，它可以根据乐谱的条件，迭代地将噪声转换为旋律谱。通过隐式优化变异约束，DiffSinger 可以被稳定地训练并产生真实的输出。


## 二、安装

- ### 1、环境依赖

  - onnxruntime >= 1.12.0

    ```shell
    # CPU
    $ pip install onnxruntime

    # GPU
    $ pip install onnxruntime-gpu
    ```

  - paddlehub >= 2.0.0  

- ### 2.安装

    - ```shell
      $ hub install diffsinger
      ```
    -  如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
      | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测
  - ### 1、命令行预测

    ```shell
    $ hub run diffsinger \
        --input_type "word" \
        --text "小酒窝长睫毛AP是你最美的记号" \
        --notes "C#4/Db4 | F#4/Gb4 | G#4/Ab4 | A#4/Bb4 F#4/Gb4 | F#4/Gb4 C#4/Db4 | C#4/Db4 | rest | C#4/Db4 | A#4/Bb4 | G#4/Ab4 | A#4/Bb4 | G#4/Ab4 | F4 | C#4/Db4" \
        --notes_duration "0.407140 | 0.376190 | 0.242180 | 0.509550 0.183420 | 0.315400 0.235020 | 0.361660 | 0.223070 | 0.377270 | 0.340550 | 0.299620 | 0.344510 | 0.283770 | 0.323390 | 0.360340" \
        --sample_num 1 \
        --save_dir "outputs"

    $ hub run diffsinger \
        --input_type "phoneme" \
        --text "感 谢 我 们 一 起 走 了 那 么 久  SP 又 再 一 次 回 到  凉 凉 深 秋  SP" \
        --ph_seq "g an x ie w o m en y i q i z ou l e n a m e j iu iu SP y ou z ai y i c i h ui d ao ao l iang l iang sh en q iu iu SP" \
        --note_seq "E4 E4 E4 E4 D4 D4 C#4 C#4 E4 E4 E4 E4 D4 D4 D4 D4 D4 D4 C#4 C#4 C#4 C#4 D4 rest D4 D4 D4 D4 E4 E4 F#4 F#4 D4 D4 G4 G4 A4 G4 G4 G4 G4 F#4 F#4 F#4 F#4 G4 rest" \
        --note_dur_seq "0.176089 0.176089 0.300448 0.300448 0.20077 0.20077 0.260772 0.260772 0.132239 0.132239 0.488848 0.488848 0.299037 0.299037 0.40449 0.40449 0.260768 0.260768 0.229999 0.229999 0.234425 0.234425 0.461538 0.257883 0.173526 0.173526 0.30519 0.30519 0.12564 0.12564 0.288726 0.288726 0.243713 0.243713 0.191862 0.191862 0.271729 0.224616 0.224616 0.390127 0.390127 0.446539 0.446539 0.296796 0.296796 0.346154 1" \
        --is_slur_seq "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0" \
        --sample_num 1 \
        --save_dir "outputs"
    ```

  - ### 2、预测代码示例

    ```python
    import paddlehub as hub

    module = hub.Module(name="diffsinger")
    results = module.singing_voice_synthesis(
      inputs={
        'text': '小酒窝长睫毛AP是你最美的记号',
        'notes': 'C#4/Db4 | F#4/Gb4 | G#4/Ab4 | A#4/Bb4 F#4/Gb4 | F#4/Gb4 C#4/Db4 | C#4/Db4 | rest | C#4/Db4 | A#4/Bb4 | G#4/Ab4 | A#4/Bb4 | G#4/Ab4 | F4 | C#4/Db4',
        'notes_duration': '0.407140 | 0.376190 | 0.242180 | 0.509550 0.183420 | 0.315400 0.235020 | 0.361660 | 0.223070 | 0.377270 | 0.340550 | 0.299620 | 0.344510 | 0.283770 | 0.323390 | 0.360340',
        'input_type': 'word'
      },
      sample_num=1,
      save_audio=True,
      save_dir='outputs'
    )
    ```

  - ### 3、API

    ```python
    def singing_voice_synthesis(
      inputs: Dict[str, str],
      sample_num: int = 1,
      save_audio: bool = True,
      save_dir: str = 'outputs'
    ) -> Dict[str, Union[List[List[int]], int]]:
    ```

    - 歌声合成 API

    - **参数**

      * inputs (Dict\[str, str\]): 输入数据，支持如下两种格式；

        ```python
        {
          'text': '小酒窝长睫毛AP是你最美的记号',
          'notes': 'C#4/Db4 | F#4/Gb4 | G#4/Ab4 | A#4/Bb4 F#4/Gb4 | F#4/Gb4 C#4/Db4 | C#4/Db4 | rest | C#4/Db4 | A#4/Bb4 | G#4/Ab4 | A#4/Bb4 | G#4/Ab4 | F4 | C#4/Db4',
          'notes_duration': '0.407140 | 0.376190 | 0.242180 | 0.509550 0.183420 | 0.315400 0.235020 | 0.361660 | 0.223070 | 0.377270 | 0.340550 | 0.299620 | 0.344510 | 0.283770 | 0.323390 | 0.360340',
          'input_type': 'word'
        }
        {
          'text': '小酒窝长睫毛AP是你最美的记号',
          'notes': 'C#4/Db4 | F#4/Gb4 | G#4/Ab4 | A#4/Bb4 F#4/Gb4 | F#4/Gb4 C#4/Db4 | C#4/Db4 | rest | C#4/Db4 | A#4/Bb4 | G#4/Ab4 | A#4/Bb4 | G#4/Ab4 | F4 | C#4/Db4',
          'notes_duration': '0.407140 | 0.376190 | 0.242180 | 0.509550 0.183420 | 0.315400 0.235020 | 0.361660 | 0.223070 | 0.377270 | 0.340550 | 0.299620 | 0.344510 | 0.283770 | 0.323390 | 0.360340',
          'input_type': 'word'
        }
        ```

      * sample_num (int): 生成音频的数量；
      * save_audio (bool): 是否保存音频文件；
      * save\_dir (str): 保存处理结果的文件目录。

    - **返回**

      * res (Dict\[str, Union\[List\[List\[int\]\], int\]\]): 歌声合成结果，一个字典，包容如下内容；

        * wavs: 歌声音频数据
        * sample_rate: 音频采样率

## 四、服务部署

- PaddleHub Serving 可以部署一个歌声合成的在线服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：

    ```shell
     $ hub serving start -m diffsinger
    ```

    - 这样就完成了一个歌声合成服务化API的部署，默认端口号为8866。

- ### 第二步：发送预测请求

  - 配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

    ```python
    import requests
    import json

    data = {
        'inputs': {
                'text': '小酒窝长睫毛AP是你最美的记号',
                'notes': 'C#4/Db4 | F#4/Gb4 | G#4/Ab4 | A#4/Bb4 F#4/Gb4 | F#4/Gb4 C#4/Db4 | C#4/Db4 | rest | C#4/Db4 | A#4/Bb4 | G#4/Ab4 | A#4/Bb4 | G#4/Ab4 | F4 | C#4/Db4',
                'notes_duration': '0.407140 | 0.376190 | 0.242180 | 0.509550 0.183420 | 0.315400 0.235020 | 0.361660 | 0.223070 | 0.377270 | 0.340550 | 0.299620 | 0.344510 | 0.283770 | 0.323390 | 0.360340',
                'input_type': 'word'
            },
        'save_audio': False,
    }
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/diffsinger"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    results = r.json()['results']
    ```

## 五、参考资料

* 论文：[DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism](https://arxiv.org/abs/2105.02446)

* 官方实现：[MoonInTheRiver/DiffSinger](https://github.com/MoonInTheRiver/DiffSinger)

## 六、更新历史

* 1.0.0

  初始发布

  ```shell
  $ hub install diffsinger==1.0.0
  ```

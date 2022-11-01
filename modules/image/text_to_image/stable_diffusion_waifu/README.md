# stable_diffusion_waifu

|模型名称|stable_diffusion_waifu|
| :--- | :---: |
|类别|多模态-文图生成|
|网络|CLIP Text Encoder+UNet+VAD|
|数据集|-|
|是否支持Fine-tuning|否|
|模型大小|4.0GB|
|最新更新日期|2022-10-17|
|数据指标|-|

## 一、模型基本信息

### 应用效果展示

  - 输入文本 "Goku"

  - 输出图像
  <p align="center">
    <img src="https://user-images.githubusercontent.com/22424850/196138387-577da86e-2910-4f7e-abe3-9ac927df7320.png"  width = "80%" hspace='10'/>
  <br />

  - 生成过程
  <p align="center">
    <img src="https://user-images.githubusercontent.com/22424850/196138374-1b3c8df7-0a35-4216-ba4f-c2bd0a0826ff.gif"  width = "80%" hspace='10'/>
  <br />

### 模型介绍

Stable Diffusion是一种潜在扩散模型(Latent Diffusion)， 属于生成类模型，这类模型通过对随机噪声进行一步步地迭代降噪并采样来获得感兴趣的图像，当前取得了令人惊艳的效果。相比于Disco Diffusion, Stable Diffusion通过在低纬度的潜在空间（lower dimensional latent space）而不是原像素空间来做迭代，极大地降低了内存和计算量的需求，并且在V100上一分钟之内即可以渲染出想要的图像，欢迎体验。本模块采用hakurei的[waifu-diffusion](https://huggingface.co/hakurei/waifu-diffusion)的预训练参数，可用于生成二次元的卡通形象。


更多详情请参考论文：[High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

## 二、安装

- ### 1、环境依赖

  - paddlepaddle >= 2.0.0

  - paddlehub >= 2.0.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install stable_diffusion_waifu
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)


## 三、模型API预测  

- ### 1、命令行预测

  - ```shell
    $ hub run stable_diffusion_waifu --text_prompts "Goku" --output_dir stable_diffusion_waifu_out
    ```

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub

    module = hub.Module(name="stable_diffusion_waifu")
    text_prompts = ["Goku"]
    # 生成图像, 默认会在stable_diffusion_waifu_out目录保存图像
    # 返回的da是一个DocumentArray对象，保存了所有的结果，包括最终结果和迭代过程的中间结果
    # 可以通过操作DocumentArray对象对生成的图像做后处理，保存或者分析
    # 您可以设置batch_size一次生成多张
    da = module.generate_image(text_prompts=text_prompts, batch_size=3, output_dir='./stable_diffusion_out/')  
    # 展示所有的中间结果
    da[0].chunks[-1].chunks.plot_image_sprites(skip_empty=True, show_index=True, keep_aspect_ratio=True)
    # 将整个生成过程保存为一个动态图gif
    da[0].chunks[-1].chunks.save_gif('stable_diffusion_waifu_out-merged-result.gif')
    # da索引的是prompt, da[0].chunks索引的是该prompt下生成的第一张图，在batch_size不为1时能同时生成多张图
    # 您也可以按照上述操作显示单张图，如第0张的生成过程
    da[0].chunks[0].chunks.plot_image_sprites(skip_empty=True, show_index=True, keep_aspect_ratio=True)
    da[0].chunks[0].chunks.save_gif('stable_diffusion_waifu_out-image-0-result.gif')
    ```

- ### 3、API

  - ```python
    def generate_image(
            text_prompts,
            width_height: Optional[List[int]] = [512, 512],
            seed: Optional[int] = None,
            batch_size: Optional[int] = 1,
            output_dir: Optional[str] = 'stable_diffusion_out'):
    ```

    - 文图生成API，生成文本描述内容的图像。

    - **参数**

      - text_prompts(str): 输入的语句，描述想要生成的图像的内容, 如卡通人物Goku。
      - width_height(Optional[List[int]]): 指定最终输出图像的宽高，宽和高都需要是64的倍数，生成的图像越大，所需要的计算时间越长。
      - seed(Optional[int]): 随机种子，由于输入默认是随机高斯噪声，设置不同的随机种子会由不同的初始输入，从而最终生成不同的结果，可以设置该参数来获得不同的输出图像。
      - batch_size(Optional[int]): 指定每个prompt一次生成的图像的数量。
      - output_dir(Optional[str]): 保存输出图像的目录，默认为"stable_diffusion_out"。


    - **返回**
      - ra(DocumentArray): DocumentArray对象， 包含`n_batches`个Documents，其中每个Document都保存了迭代过程的所有中间结果。详细可参考[DocumentArray使用文档](https://docarray.jina.ai/fundamentals/documentarray/index.html)。

## 四、服务部署

- PaddleHub Serving可以部署一个在线文图生成服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m stable_diffusion_waifu
    ```

  - 这样就完成了一个文图生成的在线服务API的部署，默认端口号为8866。

  - **NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA\_VISIBLE\_DEVICES环境变量，否则不用设置。

- ### 第二步：发送预测请求

  - 配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果，返回的预测结果在反序列化后即是上述接口声明中说明的DocumentArray类型，返回后对结果的操作方式和使用generate_image接口完全相同。

  - ```python
    import requests
    import json
    import cv2
    import base64
    from docarray import DocumentArray

    # 发送HTTP请求
    data = {'text_prompts': 'Goku'}
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/stable_diffusion_waifu"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 获取返回结果
    r.json()["results"]
    da = DocumentArray.from_base64(r.json()["results"])
    # 保存结果图
    da[0].save_uri_to_file('stable_diffusion_waifu_out.png')
    # 将生成过程保存为一个动态图gif
    da[0].chunks[0].chunks.save_gif('stable_diffusion_waifu_out.gif')
    ```

## 五、更新历史

* 1.0.0

  初始发布

  ```shell
  $ hub install stable_diffusion_waifu == 1.0.0
  ```

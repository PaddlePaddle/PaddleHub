# stable_diffusion

|模型名称|stable_diffusion|
| :--- | :---: |
|类别|多模态-文图生成|
|网络|CLIP Text Encoder+UNet+VAD|
|数据集|-|
|是否支持Fine-tuning|否|
|模型大小|4.0GB|
|最新更新日期|2022-08-26|
|数据指标|-|

## 一、模型基本信息

### 应用效果展示

  - 输入文本 "in the morning light,Overlooking TOKYO city by greg rutkowski and thomas kinkade,Trending on artstation."

  - 输出图像
  <p align="center">
    <img src="https://user-images.githubusercontent.com/22424850/186873437-2e426acd-7656-4d37-9ee4-8cafa48f097f.png"  width = "80%" hspace='10'/>
  <br />

  - 生成过程
  <p align="center">
    <img src="https://user-images.githubusercontent.com/22424850/186873216-d2a9761a-78b0-4f6a-97ec-919768f324f5.gif"  width = "80%" hspace='10'/>
  <br />

### 模型介绍

Stable Diffusion是一种潜在扩散模型(Latent Diffusion)， 属于生成类模型，这类模型通过对随机噪声进行一步步地迭代降噪并采样来获得感兴趣的图像，当前取得了令人惊艳的效果。相比于Disco Diffusion, Stable Diffusion通过在低纬度的潜在空间（lower dimensional latent space）而不是原像素空间来做迭代，极大地降低了内存和计算量的需求，并且在V100上一分钟之内即可以渲染出想要的图像，欢迎体验。

更多详情请参考论文：[High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

## 二、安装

- ### 1、环境依赖

  - paddlepaddle >= 2.0.0

  - paddlehub >= 2.0.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install stable_diffusion
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)


## 三、模型API预测  

- ### 1、命令行预测

  - ```shell
    $ hub run stable_diffusion --text_prompts "in the morning light,Overlooking TOKYO city by greg rutkowski and thomas kinkade,Trending on artstation." --output_dir stable_diffusion_out
    ```

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub

    module = hub.Module(name="stable_diffusion")
    text_prompts = ["in the morning light,Overlooking TOKYO city by greg rutkowski and thomas kinkade,Trending on artstation."]
    # 生成图像, 默认会在stable_diffusion_out目录保存图像
    # 返回的da是一个DocumentArray对象，保存了所有的结果，包括最终结果和迭代过程的中间结果
    # 可以通过操作DocumentArray对象对生成的图像做后处理，保存或者分析
    # 您可以设置n_batches一次生成多张
    da = module.generate_image(text_prompts=text_prompts, n_batches=3, output_dir='./stable_diffusion_out/')  
    # 展示所有的中间结果
    da[0].chunks[-1].chunks.plot_image_sprites(skip_empty=True, show_index=True, keep_aspect_ratio=True)
    # 将整个生成过程保存为一个动态图gif
    da[0].chunks[-1].chunks.save_gif('stable_diffusion_out-merged-result.gif')
    # da索引的是prompt, da[0].chunks索引的是该prompt下生成的第一张图，在n_batches不为1时能同时生成多张图
    # 您也可以按照上述操作显示单张图，如第0张的生成过程
    da[0].chunks[0].chunks.plot_image_sprites(skip_empty=True, show_index=True, keep_aspect_ratio=True)
    da[0].chunks[0].chunks.save_gif('stable_diffusion_out-image-0-result.gif')
    ```

- ### 3、API

  - ```python
    def generate_image(
            text_prompts,
            style: Optional[str] = None,
            artist: Optional[str] = None,
            width_height: Optional[List[int]] = [512, 512],
            seed: Optional[int] = None,
            batch_size: Optional[int] = 1,
            output_dir: Optional[str] = 'stable_diffusion_out'):
    ```

    - 文图生成API，生成文本描述内容的图像。

    - **参数**

      - text_prompts(str): 输入的语句，描述想要生成的图像的内容。通常比较有效的构造方式为 "一段描述性的文字内容" + "指定艺术家的名字"，如"in the morning light,Overlooking TOKYO city by greg rutkowski and thomas kinkade,Trending on artstation."。prompt的构造可以参考[网站](https://docs.google.com/document/d/1XUT2G9LmkZataHFzmuOtRXnuWBfhvXDAo8DkS--8tec/edit#)。
      - style(Optional[str]): 指定绘画的风格，如'watercolor','Chinese painting'等。当不指定时，风格完全由您所填写的prompt决定。
      - artist(Optional[str]): 指定特定的艺术家，如Greg Rutkowsk、krenz，将会生成所指定艺术家的绘画风格。当不指定时，风格完全由您所填写的prompt决定。各种艺术家的风格可以参考[网站](https://weirdwonderfulai.art/resources/disco-diffusion-70-plus-artist-studies/)。
      - width_height(Optional[List[int]]): 指定最终输出图像的宽高，宽和高都需要是64的倍数，生成的图像越大，所需要的计算时间越长。
      - seed(Optional[int]): 随机种子，由于输入默认是随机高斯噪声，设置不同的随机种子会由不同的初始输入，从而最终生成不同的结果，可以设置该参数来获得不同的输出图像。
      - batch_size(Optional[int]): 指定每个prompt一次生成的图像的数量。
      - output_dir(Optional[str]): 保存输出图像的目录，默认为"stable_diffusion_out"。


    - **返回**
      - ra(DocumentArray): DocumentArray对象， 包含`n_batches`个Documents，其中每个Document都保存了迭代过程的所有中间结果。详细可参考[DocumentArray使用文档](https://docarray.jina.ai/fundamentals/documentarray/index.html)。


## 四、更新历史

* 1.0.0

  初始发布

  ```shell
  $ hub install stable_diffusion == 1.0.0
  ```

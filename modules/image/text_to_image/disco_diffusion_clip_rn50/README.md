# disco_diffusion_clip_rn50

|模型名称|disco_diffusion_clip_rn50|
| :--- | :---: |
|类别|多模态-文图生成|
|网络|dd+clip RN50|
|数据集|-|
|是否支持Fine-tuning|否|
|模型大小|2.8GB|
|最新更新日期|2022-08-02|
|数据指标|-|

## 一、模型基本信息

### 应用效果展示

  - 输入文本 "A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade, Trending on artstation."

  - 输出图像
  <p align="center">
    <img src="https://user-images.githubusercontent.com/22424850/182298446-7feb530b-62cc-4e3f-a693-249ec8383daa.png"  width = "80%" hspace='10'/>
  <br />

  - 生成过程
  <p align="center">
    <img src="https://user-images.githubusercontent.com/22424850/182298453-9a8a8336-66e6-4adb-a46f-7a0fa211b467.gif"  width = "80%" hspace='10'/>
  <br />

### 模型介绍

disco_diffusion_clip_rn50 是一个文图生成模型，可以通过输入一段文字来生成符合该句子语义的图像。该模型由两部分组成，一部分是扩散模型，是一种生成模型，可以从噪声输入中重建出原始图像。另一部分是多模态预训练模型（CLIP), 可以将文本和图像表示在同一个特征空间，相近语义的文本和图像在该特征空间里距离会更相近。在该文图生成模型中，扩散模型负责从初始噪声或者指定初始图像中来生成目标图像，CLIP负责引导生成图像的语义和输入的文本的语义尽可能接近，随着扩散模型在CLIP的引导下不断的迭代生成新图像，最终能够生成文本所描述内容的图像。该模块中使用的CLIP模型结构为RN50。

更多详情请参考论文：[Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

## 二、安装

- ### 1、环境依赖

  - paddlepaddle >= 2.0.0

  - paddlehub >= 2.2.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install disco_diffusion_clip_rn50
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)


## 三、模型API预测  

- ### 1、命令行预测

  - ```shell
    $ hub run disco_diffusion_clip_rn50 --text_prompts "A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade, Trending on artstation." --output_dir disco_diffusion_clip_rn50_out
    ```

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub

    module = hub.Module(name="disco_diffusion_clip_rn50")
    text_prompts = ["A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade, Trending on artstation."]
    # 生成图像, 默认会在disco_diffusion_clip_rn50_out目录保存图像
    # 返回的da是一个DocumentArray对象，保存了所有的结果，包括最终结果和迭代过程的中间结果
    # 可以通过操作DocumentArray对象对生成的图像做后处理，保存或者分析
    da = module.generate_image(text_prompts=text_prompts, output_dir='./disco_diffusion_clip_rn50_out/')  
    # 手动将最终生成的图像保存到指定路径
    da[0].save_uri_to_file('disco_diffusion_clip_rn50_out-result.png')
    # 展示所有的中间结果
    da[0].chunks.plot_image_sprites(skip_empty=True, show_index=True, keep_aspect_ratio=True)
    # 将整个生成过程保存为一个动态图gif
    da[0].chunks.save_gif('disco_diffusion_clip_rn50_out-result.gif', show_index=True, inline_display=True, size_ratio=0.5)
    ```

- ### 3、API

  - ```python
    def generate_image(
            text_prompts:
                Optional[List[str]] = [
                    'A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade, Trending on artstation.',
                    'yellow color scheme',
                ],
            init_image: Optional[str] = None,
            width_height: Optional[List[int]] = [1280, 768],
            skip_steps: Optional[int] = 10,
            steps: Optional[int] = 250,
            init_scale: Optional[int] = 1000,
            clip_guidance_scale: Optional[int] = 5000,
            tv_scale: Optional[int] = 0,
            range_scale: Optional[int] = 150,
            sat_scale: Optional[int] = 0,
            perlin_init: Optional[bool] = False,
            perlin_mode: Optional[str] = 'mixed',
            seed: Optional[int] = None,
            display_rate: Optional[int] = 10,
            n_batches: Optional[int] = 4,
            batch_name: Optional[str] = '',
            use_gpu: Optional[bool] = True,
            output_dir: Optional[str] = 'disco_diffusion_clip_rn50_out'):
    ```

    - 文图生成API，生成文本描述内容的图像。

    - **参数**

      - text_prompts(Optional[List[str]]): 输入的语句，描述想要生成的图像的内容。
      - init_image(Optional[str]): 初始图像的路径，通常可以不需要指定初始图像，默认的初始图像为高斯噪声。
      - width_height(Optional[List[int]]): 指定最终输出图像的宽高，宽和高都需要是64的倍数，生成的图像越大，所需要的计算时间越长。
      - skip_steps(Optional[int]): 跳过的迭代次数，通常在迭代初期图像变化较快，到了迭代后期迭代变化较小，可以选择跳过末尾一定次数的迭代结束，比如如果指定了初始图像，随着迭代越靠后，越可能偏离初始图像的特征，为了让生成的图像尽可能多的保持初始图像的特征，就可以指定该参数。
      - steps(Optional[int]): 生成图像的完整迭代次数。
      - init_scale(Optional[int]): 控制生成和初始图像相似的相似度，该值在指定初始图像时候起作用，值越大，越贴近初始图像。
      - clip_guidance_scale(Optional[int]): 文本引导权重，控制生成图像和文本语义的相似度，值越大，越相似。
      - tv_scale(Optional[int]): 平滑度参数，设置为0为关闭。 值越高，越平滑。
      - range_scale(Optional[int]): 对比度参数，设置为0为关闭。 值越高，对比度越低。
      - sat_scale(Optional[int]): 饱和度参数，设置为0为关闭，值越高，饱和度越低。
      - perlin_init(Optional[bool]): 是否使用perlin噪声作为初始输入，默认为False。
      - perlin_mode(Optional[str]):  perlin噪声的模式，可选值为"colored", "gray"或者"mix"。
      - seed(Optional[int]): 随机种子，由于输入默认是随机高斯噪声，设置不同的随机种子会由不同的初始输入，从而最终生成不同的结果，可以设置该参数来获得不同的输出图像。
      - display_rate(Optional[int]): 展示并保存中间结果的间隔。例如为10，则迭代10次保存并展示一次中间结果。
      - n_batches(Optional[int]):  指定最终所生成的图像的数量。例如为2，则会生成两张不同的符合输入文本语义的图像。
      - batch_name(Optional[str]): 给定batch的名字, 该名称会构建为"reverse_diffusion-[batch_name]-seed"，作为本地生成图像的名称的前缀，并且所生成的图像默认还会上传到DocumentArray的"云剪切板"，所使用的名字也为这个。
      - use_gpu(Optional[bool]): 是否使用gpu, 默认为True。
      - output_dir(Optional[str]): 保存输出图像的目录，默认为"disco_diffusion_clip_rn50_out"。


    - **返回**
      - ra(DocumentArray): DocumentArray对象， 包含`n_batches`个Documents，其中每个Document都保存了迭代过程的所有中间结果。详细可参考[DocumentArray使用文档](https://docarray.jina.ai/fundamentals/documentarray/index.html)。

## 四、更新历史

* 1.0.0

  初始发布

  ```shell
  $ hub install disco_diffusion_clip_rn50 == 1.0.0
  ```

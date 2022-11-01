# disco_diffusion_cnclip_vitb16

|模型名称|disco_diffusion_cnclip_vitb16|
| :--- | :---: |
|类别|图像-文图生成|
|网络|dd+cnclip ViTB16|
|数据集|-|
|是否支持Fine-tuning|否|
|模型大小|2.9GB|
|最新更新日期|2022-08-02|
|数据指标|-|

## 一、模型基本信息

### 应用效果展示

   - 输入文本 "在宁静的风景中画一幅美丽的建筑画，由Arthur Adams在artstation上所作"

   - 输出图像
   <p align="center">
     <img src="https://user-images.githubusercontent.com/22424850/184838000-566f0548-f0f5-4df6-a4af-16bb70220137.png"  width = "80%" hspace='10'/>
   <br />

   - 生成过程
   <p align="center">
     <img src="https://user-images.githubusercontent.com/22424850/184837520-f30f21e1-5310-4925-a8d2-946337795010.gif"  width = "80%" hspace='10'/>
   <br />

### 模型介绍

disco_diffusion_cnclip_vitb16 是一个文图生成模型，可以通过输入一段文字来生成符合该句子语义的图像。该模型由两部分组成，一部分是扩散模型，是一种生成模型，可以从噪声输入中重建出原始图像。另一部分是多模态预训练模型（CLIP）, 可以将文本和图像表示在同一个特征空间，相近语义的文本和图像在该特征空间里距离会更相近。在该文图生成模型中，扩散模型负责从初始噪声或者指定初始图像中来生成目标图像，CLIP负责引导生成图像的语义和输入的文本的语义尽可能接近，随着扩散模型在CLIP的引导下不断的迭代生成新图像，最终能够生成文本所描述内容的图像。该模块中使用的CLIP模型结构为ViTB16。

更多详情请参考论文：[Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) 以及 [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

## 二、安装

- ### 1、环境依赖

  - paddlepaddle >= 2.0.0

  - paddlehub >= 2.2.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install disco_diffusion_cnclip_vitb16
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)


## 三、模型API预测  

- ### 1、命令行预测

  - ```shell
    $ hub run disco_diffusion_cnclip_vitb16 --text_prompts "孤舟蓑笠翁，独钓寒江雪。风格如齐白石所作。" --output_dir disco_diffusion_cnclip_vitb16_out
    ```

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub

    module = hub.Module(name="disco_diffusion_cnclip_vitb16")
    text_prompts = ["孤舟蓑笠翁，独钓寒江雪。"]
    # 生成图像, 默认会在disco_diffusion_cnclip_vitb16_out目录保存图像
    # 返回的da是一个DocumentArray对象，保存了所有的结果，包括最终结果和迭代过程的中间结果
    # 可以通过操作DocumentArray对象对生成的图像做后处理，保存或者分析
    da = module.generate_image(text_prompts=text_prompts, artist='齐白石', output_dir='./disco_diffusion_cnclip_vitb16_out/')  
    # 手动将最终生成的图像保存到指定路径
    da[0].save_uri_to_file('disco_diffusion_cnclip_vitb16_out-result.png')
    # 展示所有的中间结果
    da[0].chunks.plot_image_sprites(skip_empty=True, show_index=True, keep_aspect_ratio=True)
    # 将整个生成过程保存为一个动态图gif
    da[0].chunks.save_gif('disco_diffusion_cnclip_vitb16_out-result.gif')
    ```

- ### 3、API

  - ```python
    def generate_image(
            text_prompts,
            style: Optional[str] = None,
            artist: Optional[str] = None,
            width_height: Optional[List[int]] = [1280, 768],
            seed: Optional[int] = None,
            output_dir: Optional[str] = 'disco_diffusion_cnclip_vitb16_out'):
    ```

    - 文图生成API，生成文本描述内容的图像。

    - **参数**

      - text_prompts(str): 输入的语句，描述想要生成的图像的内容。通常比较有效的构造方式为 "一段描述性的文字内容" + "指定艺术家的名字"，如"孤舟蓑笠翁，独钓寒江雪。风格如齐白石所作"。
      - style(Optional[str]): 指定绘画的风格，如水墨画、油画、水彩画等。当不指定时，风格完全由您所填写的prompt决定。
      - artist(Optional[str]): 指定特定的艺术家，如齐白石、Greg Rutkowsk，将会生成所指定艺术家的绘画风格。当不指定时，风格完全由您所填写的prompt决定。各种艺术家的风格可以参考[网站](https://weirdwonderfulai.art/resources/disco-diffusion-70-plus-artist-studies/)。
      - width_height(Optional[List[int]]): 指定最终输出图像的宽高，宽和高都需要是64的倍数，生成的图像越大，所需要的计算时间越长。
      - seed(Optional[int]): 随机种子，由于输入默认是随机高斯噪声，设置不同的随机种子会由不同的初始输入，从而最终生成不同的结果，可以设置该参数来获得不同的输出图像。
      - output_dir(Optional[str]): 保存输出图像的目录，默认为"disco_diffusion_cnclip_vitb16_out"。


    - **返回**
      - ra(DocumentArray): DocumentArray对象， 包含`n_batches`个Documents，其中每个Document都保存了迭代过程的所有中间结果。详细可参考[DocumentArray使用文档](https://docarray.jina.ai/fundamentals/documentarray/index.html)。

## 四、服务部署

- PaddleHub Serving可以部署一个在线文图生成服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m disco_diffusion_cnclip_vitb16
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
    data = {'text_prompts': '孤舟蓑笠翁，独钓寒江雪。风格如齐白石所作'}
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/disco_diffusion_cnclip_vitb16"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 获取返回结果
    da = DocumentArray.from_base64(r.json()["results"])
    # 手动将最终生成的图像保存到指定路径
    da[0].save_uri_to_file('disco_diffusion_cnclip_vitb16_out-result.png')
    # 将生成过程保存为一个动态图gif
    da[0].chunks.save_gif('disco_diffusion_cnclip_vitb16_out-result.gif')


## 五、更新历史

* 1.0.0

  初始发布

  ```shell
  $ hub install disco_diffusion_cnclip_vitb16 == 1.0.0
  ```

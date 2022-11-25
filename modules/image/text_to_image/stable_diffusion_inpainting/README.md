# stable_diffusion_inpainting

|模型名称|stable_diffusion_inpainting|
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

  - 输入文本 "a cat sitting on a bench"

  - 输入图像
  <p align="center">
    <img src="https://user-images.githubusercontent.com/22424850/192967498-15458743-be08-4af0-b055-5bbe72c0b448.png"  width = "80%" hspace='10'/>
  <br />

  - 输入mask
  <p align="center">
    <img src="https://user-images.githubusercontent.com/22424850/192967504-7bc17d7d-98f9-4595-b355-76280865a4ab.png"  width = "80%" hspace='10'/>
  <br />

  - 输出图像
  <p align="center">
    <img src="https://user-images.githubusercontent.com/22424850/192966967-f7b12d1d-281e-415f-b38d-32715ab6bbb4.png"  width = "80%" hspace='10'/>
  <br />

  - 生成过程
  <p align="center">
    <img src="https://user-images.githubusercontent.com/22424850/192966945-19875111-31cc-42dd-85e0-3842a8df70d3.gif"  width = "80%" hspace='10'/>
  <br />

### 模型介绍

Stable Diffusion是一种潜在扩散模型(Latent Diffusion)， 属于生成类模型，这类模型通过对随机噪声进行一步步地迭代降噪并采样来获得感兴趣的图像，当前取得了令人惊艳的效果。相比于Disco Diffusion, Stable Diffusion通过在低纬度的潜在空间（lower dimensional latent space）而不是原像素空间来做迭代，极大地降低了内存和计算量的需求，并且在V100上一分钟之内即可以渲染出想要的图像，欢迎体验。该模块支持输入文本以及一张图片，一张掩码图片，对掩码部分的内容进行改变。

更多详情请参考论文：[High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

## 二、安装

- ### 1、环境依赖

  - paddlepaddle >= 2.0.0

  - paddlehub >= 2.0.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install stable_diffusion_inpainting
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)


## 三、模型API预测  

- ### 1、命令行预测

  - ```shell
    $ hub run stable_diffusion_inpainting --text_prompts "a cat sitting on a bench" --init_image /PATH/TO/IMAGE --mask_image /PATH/TO/IMAGE --output_dir stable_diffusion_inpainting_out
    ```

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub

    module = hub.Module(name="stable_diffusion_inpainting")
    text_prompts = ["a cat sitting on a bench"]
    # 生成图像, 默认会在stable_diffusion_inpainting_out目录保存图像
    # 返回的da是一个DocumentArray对象，保存了所有的结果，包括最终结果和迭代过程的中间结果
    # 可以通过操作DocumentArray对象对生成的图像做后处理，保存或者分析
    # 您可以设置batch_size一次生成多张
    da = module.generate_image(text_prompts=text_prompts, init_image='/PATH/TO/IMAGE', mask_image='/PATH/TO/IMAGE', batch_size=2, output_dir='./stable_diffusion_inpainting_out/')  
    # 展示所有的中间结果
    da[0].chunks[-1].chunks.plot_image_sprites(skip_empty=True, show_index=True, keep_aspect_ratio=True)
    # 将整个生成过程保存为一个动态图gif
    da[0].chunks[-1].chunks.save_gif('stable_diffusion_inpainting_out-merged-result.gif')
    # da索引的是prompt, da[0].chunks索引的是该prompt下生成的第一张图，在batch_size不为1时能同时生成多张图
    # 您也可以按照上述操作显示单张图，如第0张的生成过程
    da[0].chunks[0].chunks.plot_image_sprites(skip_empty=True, show_index=True, keep_aspect_ratio=True)
    da[0].chunks[0].chunks.save_gif('stable_diffusion_inpainting-image-0-result.gif')
    ```

- ### 3、API

  - ```python
    def generate_image(
            text_prompts,
            init_image,
            mask_image,
            strength: float = 0.8,
            width_height: Optional[List[int]] = [512, 512],
            seed: Optional[int] = None,
            batch_size: Optional[int] = 1,
            display_rate: Optional[int] = 5,
            output_dir: Optional[str] = 'stable_diffusion_inpainting_out'):
    ```

    - 文图生成API，生成文本描述内容的图像。

    - **参数**

      - text_prompts(str): 输入的语句，描述想要生成的图像的内容。
      - init_image(str|numpy.ndarray|PIL.Image): 输入的初始图像。
      - mask_image(str|numpy.ndarray|PIL.Image): 输入的掩码图像。
      - strength(float): 控制添加到输入图像的噪声强度，取值范围0到1。越接近1.0，图像变化越大。
      - width_height(Optional[List[int]]): 指定最终输出图像的宽高，宽和高都需要是64的倍数，生成的图像越大，所需要的计算时间越长。
      - seed(Optional[int]): 随机种子，由于输入默认是随机高斯噪声，设置不同的随机种子会由不同的初始输入，从而最终生成不同的结果，可以设置该参数来获得不同的输出图像。
      - batch_size(Optional[int]): 指定每个prompt一次生成的图像的数量。
      - display_rate(Optional[int]): 保存中间结果的频率，默认每5个step保存一次中间结果，如果不需要中间结果来让程序跑的更快，可以将这个值设大。
      - output_dir(Optional[str]): 保存输出图像的目录，默认为"stable_diffusion_out"。


    - **返回**
      - ra(DocumentArray): DocumentArray对象， 包含`n_batches`个Documents，其中每个Document都保存了迭代过程的所有中间结果。详细可参考[DocumentArray使用文档](https://docarray.jina.ai/fundamentals/documentarray/index.html)。

## 四、服务部署

- PaddleHub Serving可以部署一个在线文图生成服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m stable_diffusion_inpainting
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

    def cv2_to_base64(image):
      data = cv2.imencode('.jpg', image)[1]
      return base64.b64encode(data.tobytes())

    # 发送HTTP请求
    data = {'text_prompts': 'a cat sitting on a bench', 'init_image': cv2_to_base64(cv2.imread('/PATH/TO/IMAGE')),
            'mask_image': cv2_to_base64(cv2.imread('/PATH/TO/IMAGE')}
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/stable_diffusion_inpainting"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 获取返回结果
    r.json()["results"]
    da = DocumentArray.from_base64(r.json()["results"])
    # 保存结果图
    da[0].save_uri_to_file('stable_diffusion_inpainting_out.png')
    # 将生成过程保存为一个动态图gif
    da[0].chunks[0].chunks.save_gif('stable_diffusion_inpainting_out.gif')
    ```

## 五、更新历史

* 1.0.0

  初始发布

  ```shell
  $ hub install stable_diffusion_inpainting == 1.0.0
  ```

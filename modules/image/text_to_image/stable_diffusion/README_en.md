# stable_diffusion

|Module Name|stable_diffusion|
| :--- | :---: |
|Category|text to image|
|Network|CLIP Text Encoder+UNet+VAD|
|Dataset|-|
|Fine-tuning supported or not|No|
|Module Size|4.0GB|
|Latest update date|2022-08-26|
|Data indicators|-|

## I.Basic Information

### Application Effect Display

  - Prompt "in the morning light,Overlooking TOKYO city by greg rutkowski and thomas kinkade,Trending on artstation."

  - Output image
  <p align="center">
    <img src="https://user-images.githubusercontent.com/22424850/186873437-2e426acd-7656-4d37-9ee4-8cafa48f097f.png"  width = "80%" hspace='10'/>
  <br />

  - Generating process
  <p align="center">
    <img src="https://user-images.githubusercontent.com/22424850/186873216-d2a9761a-78b0-4f6a-97ec-919768f324f5.gif"  width = "80%" hspace='10'/>
  <br />

### Module Introduction

Stable Diffusion is a latent diffusion model (Latent Diffusion), which belongs to the generative model. This kind of model obtains the images by iteratively denoising  noise and sampling step by step, and currently has achieved amazing results. Compared with Disco Diffusion, Stable Diffusion iterates in a lower dimensional latent space instead of the original pixel space, which greatly reduces the memory and computational requirements. You can render the desired image within a minute on the V100, welcome to enjoy it in [aistudio](https://aistudio.baidu.com/aistudio/projectdetail/4512600).

For more details, please refer to [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

## II.Installation

- ### 1、Environmental Dependence

  - paddlepaddle >= 2.0.0

  - paddlehub >= 2.0.0    | [How to install PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、Installation

  - ```shell
    $ hub install stable_diffusion
    ```
  - In case of any problems during installation, please refer to: [Windows_Quickstart]() | [Linux_Quickstart]() | [Mac_Quickstart]()


## III.Module API Prediction  

- ### 1、Command line Prediction

  - ```shell
    $ hub run stable_diffusion --text_prompts "in the morning light,Overlooking TOKYO city by greg rutkowski and thomas kinkade,Trending on artstation." --output_dir stable_diffusion_out
    ```

- ### 2、Prediction Code Example

  - ```python
    import paddlehub as hub

    module = hub.Module(name="stable_diffusion")
    text_prompts = ["in the morning light,Overlooking TOKYO city by greg rutkowski and thomas kinkade,Trending on artstation."]
    # Output images will be saved in stable_diffusion_out directory.
    # The returned da is a DocumentArray object, which contains all immediate and final results
    # You can manipulate the DocumentArray object to do post-processing and save images
    # you can set batch_size parameter to generate number of batch_size images at one inference step.
    da = module.generate_image(text_prompts=text_prompts, batch_size=3, output_dir='./stable_diffusion_out/')  
    # Show all immediate results
    da[0].chunks[-1].chunks.plot_image_sprites(skip_empty=True, show_index=True, keep_aspect_ratio=True)
    # Save the generating process as a gif
    da[0].chunks[-1].chunks.save_gif('stable_diffusion_out-merged-result.gif')
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

    - Image generating api, which generates an image corresponding to your prompt.

    - **Parameters**

      - text_prompts(str): Prompt, used to describe your image content. You can construct a prompt conforms to the format "content" + "artist/style", such as "in the morning light,Overlooking TOKYO city by greg rutkowski and thomas kinkade,Trending on artstation.". For more details, you can refer to [website](https://docs.google.com/document/d/1XUT2G9LmkZataHFzmuOtRXnuWBfhvXDAo8DkS--8tec/edit#).
      - style(Optional[str]): Image style, such as "watercolor" and "Chinese painting". If not provided, style is totally up to your prompt.
      - artist(Optional[str]): Artist name, such as Greg Rutkowsk、krenz, image style is as whose works you choose. If not provided, style is totally up to your prompt.(https://weirdwonderfulai.art/resources/disco-diffusion-70-plus-artist-studies/).
      - width_height(Optional[List[int]]): The width and height of output images, should be better multiples of 64. The larger size is, the longger computation time is.
      - seed(Optional[int]): Random seed, different seeds result in different output images.
      - batch_size(Optional[int]): Number of images generated for one inference step.
      - output_dir(Optional[str]): Output directory, default is "stable_diffusion_out".


    - **Return**
      - ra(DocumentArray):  DocumentArray object， including `batch_size` Documents，each document keeps all immediate results during generation, please refer to [DocumentArray tutorial](https://docarray.jina.ai/fundamentals/documentarray/index.html) for more details.

## IV.Server Deployment

- PaddleHub Serving can deploy an online service of text-to-image.

- ### Step 1: Start PaddleHub Serving

  - Run the startup command：
  - ```shell
    $ hub serving start -m stable_diffusion
    ```

  - The servitization API is now deployed and the default port number is 8866.

  - **NOTE:**  If GPU is used for prediction, set CUDA_VISIBLE_DEVICES environment variable before the service, otherwise it need not be set.

- ### Step 2: Send a predictive request

  - With a configured server, use the following lines of code to send the prediction request and obtain the result.

  - ```python
    import requests
    import json
    import cv2
    import base64
    from docarray import DocumentArray

    # Send an HTTP request
    data = {'text_prompts': 'in the morning light,Overlooking TOKYO city by greg rutkowski and thomas kinkade,Trending on artstation.'}
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/stable_diffusion"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # Get results
    r.json()["results"]
    da = DocumentArray.from_base64(r.json()["results"])
    # Save final result image to a file
    da[0].save_uri_to_file('stable_diffusion_out.png')
    # Save the generating process as a gif
    da[0].chunks[0].chunks.save_gif('stable_diffusion_out.gif')
    ```

## V.Release Note

* 1.0.0

  First release

  ```shell
  $ hub install stable_diffusion == 1.0.0
  ```

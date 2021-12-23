# porn_detection_gru

|     Module Name      |  porn_detection_gru  |
|  :------------------ | :------------: |
|       Category       | text-text_review  |
|         Network      |      GRU      |
|         Dataset      | Dataset built by Baidu |
| Fine-tuning supported or not |      No       |
|     Module Size      |       20M       |
| Latest update date   |   2021-02-26   |
|   Data indicators    |       -        |

## I. Basic Information of Module

- ### Module Introduction
  - Pornography detection model can automatically distinguish whether the text is pornographic or not and give the corresponding confidence, and identify the pornographic description, vulgar communication and filthy text in the text.
  - porn_detection_gru adopts GRU network structure and cuts words according to word granularity, which has high prediction speed. The maximum sentence length of this model is 256 words, and only prediction is supported.


## II. Installation

- ### 1、Environmental dependence

  - paddlepaddle >= 1.6.2
  
  - paddlehub >= 1.6.0    | [How to install PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、Installation

  - ```shell
    $ hub install porn_detection_gru
    ```
  - If you have problems during installation, please refer to:[windows_quickstart](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [linux_quickstart](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [mac_quickstart](../../../../docs/docs_ch/get_start/mac_quickstart.md)



## III. Module API and Prediction

- ### 1、Command line Prediction

  - ```shell
    $ hub run porn_detection_gru --input_text "黄片下载"
    ```
    
  - or

  - ```shell
    $ hub run porn_detection_gru --input_file test.txt
    ```
    
    - test.txt stores the text to be reviewed. Each line contains only one text
    
  - If you want to call the Hub module through the command line, please refer to: [PaddleHub Command line instruction](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、Prediction Code Example

  - ```python
    import paddlehub as hub
    
    porn_detection_gru = hub.Module(name="porn_detection_gru")
    
    test_text = ["黄片下载", "打击黄牛党"]
    
    results = porn_detection_gru.detection(texts=test_text, use_gpu=True, batch_size=1)   # If you do not use GPU, please set use_gpu=False
    
    for index, text in enumerate(test_text):
        results[index]["text"] = text
    for index, result in enumerate(results):
        print(results[index])
    
    # The output：
    # {'text': '黄片下载', 'porn_detection_label': 1, 'porn_detection_key': 'porn', 'porn_probs': 0.9324, 'not_porn_probs': 0.0676}
    # {'text': '打击黄牛党', 'porn_detection_label': 0, 'porn_detection_key': 'not_porn', 'porn_probs': 0.0004, 'not_porn_probs': 0.9996}
    ```

  
- ### 3、API

  - ```python
    def detection(texts=[], data={}, use_gpu=False, batch_size=1)
    ```
  
    - prediction api of porn_detection_gru，to identify whether input sentences contain pornography

    - **Parameter**

      - texts(list): data to be predicted, if texts parameter is used, there is no need to pass in data parameter. You can use any of the two parameters.

      - data(dict): predicted data , key must be text，value is data to be predicted. if data parameter is used, there is no need to pass in texts parameter. You can use any of the two parameters. It is suggested to use texts parameter, and data parameter will be discarded later.

      - use_gpu(bool): use GPU or not. If GPU is used for prediction, set CUDA_VISIBLE_DEVICES environment variable before prediction. Otherwise, need not set it.

    - **Return**

      - results(list): prediction result


  - ```python
    def get_labels()
    ```
    - get the category of porn_detection_gru

    - **Return**

      - labels(dict): the category of porn_detection_gru (Dichotomies, yes/no)

  - ```python
    def get_vocab_path()
    ```

    - get a vocabulary for pre-training

    - **Return**

      - vocab_path(str): Vocabulary path



## IV. Server Deployment

- PaddleHub Serving can deploy an online pornography detection service and you can use this interface for online Web applications.

- ## Step 1: Start PaddleHub Serving

  - Run the startup command:
  - ```shell
    $ hub serving start -m porn_detection_gru  
    ```

  - The model loading process is displayed on startup. After the startup is successful, the following information is displayed:
  - ```shell
    Loading porn_detection_gur successful.
    ```

  - The servitization API is now deployed and the default port number is 8866.

  - **NOTE:**  If GPU is used for prediction, set CUDA_VISIBLE_DEVICES environment variable before prediction. Otherwise, need not set it.


- ## Step 2: Send a predictive request

  - After configuring the server, the following lines of code can be used to send the prediction request and obtain the prediction result
  - ```python
    import requests
    import json
    
    # data to be predicted
    text = ["黄片下载", "打击黄牛党"]

    # Set the running configuration
    # Corresponding local forecast porn_detection_gru.detection(texts=text, batch_size=1, use_gpu=True)
    data = {"texts": text, "batch_size": 1, "use_gpu":True}

    # set the prediction method to porn_detection_gru and send a POST request, content-type should be set to json
    # HOST_IP is the IP address of the server
    url = "http://HOST_IP:8866/predict/porn_detection_gru"
    headers = {"Content-Type": "application/json"}
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # print prediction result
    print(json.dumps(r.json(), indent=4, ensure_ascii=False))
    ```

  - For more information about PaddleHub Serving, please refer to:[Serving Deployment](../../../../docs/docs_ch/tutorial/serving.md)




## V. Release Note

* 1.0.0

  First release

* 1.1.0

  Improves prediction performance and simplifies interface usage

  - ```shell
    $ hub install porn_detection_gru==1.1.0
    ```
    

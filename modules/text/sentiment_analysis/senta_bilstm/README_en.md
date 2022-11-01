# senta_bilstm

|     Module Name      |  senta_bilstm  |
|  :------------------ | :------------: |
|       Category       | text-sentiment_analysis  |
|         Network      |      BiLSTM      |
|         Dataset      | Dataset built by Baidu |
| Fine-tuning supported or not |      No       |
|     Module Size      |       690M       |
| Latest update date   |   2021-02-26   |
|   Data indicators    |       -        |


## I. Basic Information of Module

- ### Module Introduction

  - Sentiment Classification (Senta for short) can automatically judge the emotional polarity category of Chinese texts with subjective description and give corresponding confidence, which can help enterprises understand users' consumption habits, analyze hot topics and crisis public opinion monitoring, and provide favorable decision support for enterprises. The model is based on a bidirectional LSTM structure, with positive and negative emotion types.



## II. Installation

- ### 1、Environmental dependence

  - paddlepaddle >= 1.8.0

  - paddlehub >= 1.8.0    | [How to install PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、Installation

  - ```shell
    $ hub install senta_bilstm
    ```
  - If you have problems during installation, please refer to:[windows_quickstart](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [linux_quickstart](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [mac_quickstart](../../../../docs/docs_ch/get_start/mac_quickstart.md)



## III. Module API and Prediction

- ### 1、Command line Prediction

  - ```shell
    $ hub run senta_bilstm --input_text "这家餐厅很好吃"
    ```
    or
  - ```shell
    $ hub run senta_bilstm --input_file test.txt
    ```  
    - test.txt stores the text to be predicted, for example:

      > 这家餐厅很好吃

      > 这部电影真的很差劲

    - If you want to call the Hub module through the command line, please refer to: [PaddleHub Command line instruction](../../../../docs/docs_ch/tutorial/cmd_usage.rst)


- ### 2、Prediction Code Example

  - ```python
    import paddlehub as hub

    senta = hub.Module(name="senta_bilstm")
    test_text = ["这家餐厅很好吃", "这部电影真的很差劲"]
    results = senta.sentiment_classify(texts=test_text,
                                       use_gpu=False,
                                       batch_size=1)

    for result in results:
        print(result['text'])
        print(result['sentiment_label'])
        print(result['sentiment_key'])
        print(result['positive_probs'])
        print(result['negative_probs'])

    # 这家餐厅很好吃 1 positive 0.9407 0.0593
    # 这部电影真的很差劲 0 negative 0.02 0.98
    ```

- ### 3、API

  - ```python
    def sentiment_classify(texts=[], data={}, use_gpu=False, batch_size=1)
    ```

    - senta_bilstm predicting interfaces, predicting sentiment classification of input sentences (dichotomies, positive/negative)

    - **Parameter**

      - texts(list):  data to be predicted, if texts parameter is used, there is no need to pass in data parameter. You can use any of the two parameters.
      - data(dict): predicted data , key must be text，value is data to be predicted. if data parameter is used, there is no need to pass in texts parameter. You can use any of the two parameters. It is suggested to use texts parameter, and data parameter will be discarded later.
      - use_gpu(bool): use GPU or not. If GPU is used for prediction, set CUDA_VISIBLE_DEVICES environment variable before prediction. Otherwise, need not set it.
      - batch_size(int): batch size

    - **Return**

      - results(list): result of sentiment classification


  - ```python
    def get_labels()
    ```
    - get the category of senta_bilstm

    - **Return**

      - labels(dict): the category of senta_bilstm(Dichotomies, positive/negative)

  - ```python
    def get_vocab_path()
    ```
    - Get a vocabulary for pre-training

    - **Return**

      - vocab_path(str): Vocabulary path



## IV. Server Deployment

- PaddleHub Serving can deploy an online sentiment analysis detection service and you can use this interface for online Web applications.

- ## Step 1: Start PaddleHub Serving

  - Run the startup command:
  - ```shell
    $ hub serving start -m senta_bilstm  
    ```

  - The model loading process is displayed on startup. After the startup is successful, the following information is displayed:
  - ```shell
    Loading senta_bilstm successful.
    ```

  - The servitization API is now deployed and the default port number is 8866.

   - **NOTE:**  If GPU is used for prediction, set CUDA_VISIBLE_DEVICES environment variable before prediction. Otherwise, need not set it.

- ## Step 2: Send a predictive request

  - After configuring the server, the following lines of code can be used to send the prediction request and obtain the prediction result

  - ```python
    import requests
    import json

    # data to be predicted
    text = ["这家餐厅很好吃", "这部电影真的很差劲"]

    # Set the running configuration
    # Corresponding to local prediction senta_bilstm.sentiment_classify(texts=text, batch_size=1, use_gpu=True)
    data = {"texts": text, "batch_size": 1, "use_gpu":True}

    # set the prediction method to senta_bilstm and send a POST request, content-type should be set to json
    # HOST_IP is the IP address of the server
    url = "http://HOST_IP:8866/predict/senta_bilstm"
    headers = {"Content-Type": "application/json"}
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # print prediction result
    print(json.dumps(r.json(), indent=4, ensure_ascii=False))
    ```

  - For more information about PaddleHub Serving, please refer to:[Serving Deployment](../../../../docs/docs_ch/tutorial/serving.md)



## V. Release Note

* 1.0.0

  First release

* 1.0.1

  Vocabulary upgrade

* 1.1.0

  Significantly improve predictive performance

* 1.2.1

  Remove fluid api

  - ```shell
    $ hub install senta_bilstm==1.2.1
    ```

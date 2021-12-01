# fasttext_crawl_target_word-word_dim300_en
|Module Name|fasttext_crawl_target_word-word_dim300_en|
| :--- | :---: | 
|Category|Word Embedding|
|Network|fasttext|
|Dataset|crawl|
|Fine-tuning supported|No|
|Module Size|1.19GB|
|Vocab Size|2,000,002|
|Last update date|26 Feb, 2021|
|Data Indicators|-|

## I. Basic Information

- ### Module Introduction

    - PaddleHub provides several open source pretrained word embedding models. These embedding models are distinguished by the corpus, training methods and word embedding dimensions. For more informations, please refer to: [Summary of embedding models](https://github.com/PaddlePaddle/models/blob/release/2.0-beta/PaddleNLP/docs/embeddings.md)

## II. Installation

- ### 1. Environmental Dependence

  - paddlepaddle >= 2.0.0

  - paddlehub >= 2.0.0    | [PaddleHub Installation Guide](../../../../docs/docs_ch/get_start/installation_en.rst)

- ### 2. Installation

  - ```shell
    $ hub install fasttext_crawl_target_word-word_dim300_en
    ```

  - In case of any problems during installation, please refer to: [Windows_Quickstart](../../../../docs/docs_ch/get_start/windows_quickstart_en.md) | [Linux_Quickstart](../../../../docs/docs_ch/get_start/linux_quickstart_en.md) | [Mac_Quickstart](../../../../docs/docs_ch/get_start/mac_quickstart_en.md)

## III. Module API Prediction

- ### 1. Prediction Code Example

  - ```
    import paddlehub as hub
    embedding = hub.Module(name='fasttext_crawl_target_word-word_dim300_en')

    # Get the embedding of the word
    embedding.search("中国")
    # Calculate the cosine similarity of two word vectors
    embedding.cosine_sim("中国", "美国")
    # Calculate the inner product of two word vectors
    embedding.dot("中国", "美国")
    ```

- ### 2、API

  - ```python
    def __init__(
        *args,
        **kwargs
    )
    ```

    - Construct an embedding module object without parameters by default.

    - **Parameters**
      - `*args`： Arguments specified by the user.
      - `**kwargs`：Keyword arguments specified by the user.

    - More info[paddlenlp.embeddings](https://github.com/PaddlePaddle/models/tree/release/2.0-beta/PaddleNLP/paddlenlp/embeddings)


  - ```python
    def search(
        words: Union[List[str], str, int],
    )
    ```

    - Return the embedding of one or multiple words. The input data type can be `str`, `List[str]` and `int`, represent word, multiple words and the embedding of specified word id accordingly. Word id is related to the model vocab, vocab can be obtained by the attribute of `vocab`.

    - **参数**
      - `words`： input words or word id.


  - ```python
    def cosine_sim(
        word_a: str,
        word_b: str,
    )
    ```

    - Cosine similarity calculation. `word_a` and `word_b` should be in the voacb, or they will be replaced by `unknown_token`. 

    - **参数**
      - `word_a`： input word a.
      - `word_b`： input word b.


  - ```python
    def dot(
        word_a: str,
        word_b: str,
    )
    ```

    - Inner product calculation. `word_a` and `word_b` should be in the voacb, or they will be replaced by `unknown_token`. 

    - **参数**
      - `word_a`： input word a.
      - `word_b`： input word b.


  - ```python
    def get_vocab_path()
    ```

    - Get the path of the local vocab file.


  - ```python
    def get_tokenizer(*args, **kwargs)
    ```

    - Get the tokenizer of current model, it will return an instance of JiebaTokenizer, only supports the chinese embedding models currently.

    - **参数**
      - `*args`: Arguments specified by the user.
      - `**kwargs`: Keyword arguments specified by the user.
    
    - For more information about the arguments, please refer to[paddlenlp.data.tokenizer.JiebaTokenizer](https://github.com/PaddlePaddle/models/blob/release/2.0-beta/PaddleNLP/paddlenlp/data/tokenizer.py)

  - For more information about the usage, please refer to[paddlenlp.embeddings](https://github.com/PaddlePaddle/models/tree/release/2.0-beta/PaddleNLP/paddlenlp/embeddings)


## IV. Server Deployment

- PaddleHub Serving can deploy an online service of cosine similarity calculation.

- ### Step 1: Start PaddleHub Serving

  - Run the startup command:

  - ```shell
    $ hub serving start -m fasttext_crawl_target_word-word_dim300_en
    ```

  - The servitization API is now deployed and the default port number is 8866.

  - **NOTE:** If GPU is used for prediction, set `CUDA_VISIBLE_DEVICES` environment variable before the service, otherwise it need not be set.

- ### Step 2: Send a predictive request

  - With a configured server, use the following lines of code to send the prediction request and obtain the result

  - ```python
    import requests
    import json

    # Specify the word pairs used to calculate the cosine similarity [[word_a, word_b], [word_a, word_b], ... ]]
    word_pairs = [["中国", "美国"], ["今天", "明天"]]
    data = {"data": word_pairs}
    # Send an HTTP request
    url = "http://127.0.0.1:8866/predict/fasttext_crawl_target_word-word_dim300_en"
    headers = {"Content-Type": "application/json"}

    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    print(r.json())
    ```


## V. Release Note

* 1.0.0

  First release

* 1.0.1

  Model optimization
  - ```shell
    $ hub install fasttext_crawl_target_word-word_dim300_en==1.0.1
    ```
import os
import paddlehub as hub


def infer_with_input_text():
    # get senta module
    senta = hub.Module(name="senta")

    test_text = ["这家餐厅很好吃", "这部电影真的很差劲"]

    # get the input keys for signature 'sentiment_classify'
    data_format = senta.processor.data_format(sign_name='sentiment_classify')
    key = list(data_format.keys())[0]

    # set input dict
    input_dict = {key: test_text}

    # execute predict and print the result
    results = senta.sentiment_classify(data=input_dict)
    for index, result in enumerate(results):
        hub.logger.info("sentence %d segmented result: %s" %
                        (index + 1, result['sentiment_key']))


def infer_with_input_file():
    # get senta module
    senta = hub.Module(name="senta")

    # get the input keys for signature 'sentiment_classify'
    data_format = senta.processor.data_format(sign_name='sentiment_classify')
    key = list(data_format.keys())[0]

    # parse input file
    test_file = os.path.join("test", "test.txt")
    test_text = hub.io.parser.txt_parser.parse(test_file)

    # set input dict
    input_dict = {key: test_text}
    results = senta.sentiment_classify(data=input_dict)
    for index, result in enumerate(results):
        hub.logger.info("sentence %d segmented result: %s" %
                        (index + 1, result['sentiment_key']))


if __name__ == "__main__":
    infer_with_input_text()

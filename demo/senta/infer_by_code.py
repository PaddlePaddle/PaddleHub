import os
import paddle_hub as hub


def infer_with_input_text():
    # get senta module
    senta = hub.Module(module_dir="hub_module_senta")

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
    senta = hub.Module(module_dir="hub_module_senta")

    # get the input keys for signature 'sentiment_classify'
    data_format = senta.processor.data_format(sign_name='sentiment_classify')
    key = list(data_format.keys())[0]

    # parse input file
    test_csv = os.path.join("resources", "test", "test.csv")
    test_text = hub.io.reader.csv_reader.read(test_csv)["TEXT_INPUT"]

    # set input dict
    input_dict = {key: test_text}
    results = senta.sentiment_classify(data=input_dict)
    for index, result in enumerate(results):
        hub.logger.info("sentence %d segmented result: %s" %
                        (index + 1, result['sentiment_key']))


if __name__ == "__main__":
    infer_with_input_text()

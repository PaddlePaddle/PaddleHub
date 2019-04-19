import os
import paddlehub as hub


def infer_with_input_text():
    # get lac module
    lac = hub.Module(name="lac")

    test_text = ["今天是个好日子", "天气预报说今天要下雨", "下一班地铁马上就要到了"]

    # get the input keys for signature 'lexical_analysis'
    data_format = lac.processor.data_format(sign_name='lexical_analysis')
    key = list(data_format.keys())[0]

    # set input dict
    input_dict = {key: test_text}

    # execute predict and print the result
    results = lac.lexical_analysis(data=input_dict)
    for index, result in enumerate(results):
        hub.logger.info(
            "sentence %d segmented result: %s" % (index + 1, result['word']))


def infer_with_input_file():
    # get lac module
    lac = hub.Module(name="lac")

    # get the input keys for signature 'lexical_analysis'
    data_format = lac.processor.data_format(sign_name='lexical_analysis')
    key = list(data_format.keys())[0]

    # parse input file
    test_file = os.path.join("test", "test.txt")
    test_text = hub.io.parser.txt_parser.parse(test_file)

    # set input dict
    input_dict = {key: test_text}
    results = lac.lexical_analysis(data=input_dict)
    for index, result in enumerate(results):
        hub.logger.info(
            "sentence %d segmented result: %s" % (index + 1, result['word']))


if __name__ == "__main__":
    infer_with_input_file()

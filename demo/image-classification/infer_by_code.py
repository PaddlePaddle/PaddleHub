import os
import paddlehub as hub


def infer_with_input_text():
    # get ResNet50 module
    resnet50 = hub.Module(module_dir="ResNet50.hub_module")

    test_img_path = os.path.join("resources", "test", "test_img_bird.jpg")

    # get the input keys for signature 'classification'
    data_format = resnet50.processor.data_format(sign_name='classification')
    key = list(data_format.keys())[0]

    # set input dict
    input_dict = {key: [test_img_path]}

    # execute predict and print the result
    results = resnet50.classification(data=input_dict)
    for result in results:
        hub.logger.info(result)


def infer_with_input_file():
    # get ResNet50 module
    resnet50 = hub.Module(module_dir="ResNet50.hub_module")

    # get the input keys for signature 'classification'
    data_format = resnet50.processor.data_format(sign_name='classification')
    key = list(data_format.keys())[0]

    # parse input file
    test_file = os.path.join("resources", "test", "test.txt")
    test_images = hub.io.parser.txt_parser.parse(test_file)

    # set input dict
    input_dict = {key: test_images}
    config = {'top_only': True}
    results = resnet50.classification(data=input_dict, **config)
    for result in results:
        hub.logger.info(result)


if __name__ == "__main__":
    infer_with_input_file()

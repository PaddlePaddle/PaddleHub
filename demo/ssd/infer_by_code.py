import os
import paddlehub as hub


def infer_with_input_path():
    # get ssd module
    ssd = hub.Module(name="ssd_mobilenet_v1_pascal")

    test_img_path = os.path.join("test", "test_img_bird.jpg")

    # get the input keys for signature 'object_detection'
    data_format = ssd.processor.data_format(sign_name='object_detection')
    key = list(data_format.keys())[0]

    # set input dict
    input_dict = {key: [test_img_path]}

    # execute predict and print the result
    results = ssd.object_detection(data=input_dict)
    for result in results:
        hub.logger.info(result)


def infer_with_input_file():
    # get ssd module
    ssd = hub.Module(name="ssd_mobilenet_v1_pascal")

    # get the input keys for signature 'object_detection'
    data_format = ssd.processor.data_format(sign_name='object_detection')
    key = list(data_format.keys())[0]

    # parse input file
    test_file = os.path.join("test", "test.txt")
    test_images = hub.io.parser.txt_parser.parse(test_file)

    # set input dict
    input_dict = {key: test_images}
    results = ssd.object_detection(data=input_dict)
    for result in results:
        hub.logger.info(result)


if __name__ == "__main__":
    infer_with_input_file()

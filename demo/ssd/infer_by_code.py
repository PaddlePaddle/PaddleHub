import os
import paddlehub as hub


def infer_with_input_text():
    # get ssd module
    ssd = hub.Module(module_dir="hub_module_ssd")

    test_img_path = os.path.join("resources", "test", "test_img_bird.jpg")

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
    ssd = hub.Module(module_dir="hub_module_ssd")

    # get the input keys for signature 'object_detection'
    data_format = ssd.processor.data_format(sign_name='object_detection')
    key = list(data_format.keys())[0]

    # parse input file
    test_csv = os.path.join("resources", "test", "test.csv")
    test_images = hub.io.reader.csv_reader.read(test_csv)["IMAGE_PATH"]

    # set input dict
    input_dict = {key: test_images}
    results = ssd.object_detection(data=input_dict)
    for result in results:
        hub.logger.info(result)


if __name__ == "__main__":
    infer_with_input_file()

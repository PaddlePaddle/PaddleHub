import paddlehub as hub
from paddleocr.ppocr.utils.logging import get_logger
from paddleocr.tools.infer.utility import base64_to_cv2
from paddlehub.module.module import moduleinfo, runnable, serving


@moduleinfo(
    name="tamil_ocr_db_crnn_mobile",
    version="1.0.0",
    summary="ocr service",
    author="PaddlePaddle",
    type="cv/text_recognition")
class TamilOCRDBCRNNMobile:
    def __init__(self,
                 det=True,
                 rec=True,
                 use_angle_cls=False,
                 enable_mkldnn=False,
                 use_gpu=False,
                 box_thresh=0.6,
                 angle_classification_thresh=0.9):
        """
        initialize with the necessary elements
        Args:
            det(bool): Whether to use text detector.
            rec(bool): Whether to use text recognizer.
            use_angle_cls(bool): Whether to use text orientation classifier.
            enable_mkldnn(bool): Whether to enable mkldnn.
            use_gpu (bool): Whether to use gpu.
            box_thresh(float): the threshold of the detected text box's confidence
            angle_classification_thresh(float): the threshold of the angle classification confidence
        """
        self.logger = get_logger()
        self.model = hub.Module(
            name="multi_languages_ocr_db_crnn",
            lang="ta",
            det=det,
            rec=rec,
            use_angle_cls=use_angle_cls,
            enable_mkldnn=enable_mkldnn,
            use_gpu=use_gpu,
            box_thresh=box_thresh,
            angle_classification_thresh=angle_classification_thresh)
        self.model.name = self.name

    def recognize_text(self, images=[], paths=[], output_dir='ocr_result', visualization=False):
        """
        Get the text in the predicted images.
        Args:
            images (list(numpy.ndarray)): images data, shape of each is [H, W, C]. If images not paths
            paths (list[str]): The paths of images. If paths not images
            output_dir (str): The directory to store output images.
            visualization (bool): Whether to save image or not.
        Returns:
            res (list): The result of text detection box and save path of images.
        """
        all_results = self.model.recognize_text(
            images=images, paths=paths, output_dir=output_dir, visualization=visualization)
        return all_results

    @serving
    def serving_method(self, images, **kwargs):
        """
        Run as a service.
        """
        images_decode = [base64_to_cv2(image) for image in images]
        results = self.recognize_text(images_decode, **kwargs)
        return results

    @runnable
    def run_cmd(self, argvs):
        """
        Run as a command
        """
        results = self.model.run_cmd(argvs)
        return results

    def export_onnx_model(self, dirname: str, input_shape_dict=None, opset_version=10):
        '''
        Export the model to ONNX format.

        Args:
            dirname(str): The directory to save the onnx model.
            input_shape_dict: dictionary ``{ input_name: input_value }, eg. {'x': [-1, 3, -1, -1]}``
            opset_version(int): operator set
        '''
        self.model.export_onnx_model(dirname=dirname, input_shape_dict=input_shape_dict, opset_version=opset_version)

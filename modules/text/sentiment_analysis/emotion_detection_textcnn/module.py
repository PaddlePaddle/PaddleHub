import math
import os

import paddlehub as hub
from .processor import load_vocab
from .processor import postprocess
from .processor import preprocess
from paddlehub.module.module import moduleinfo
from paddlehub.module.module import serving


@moduleinfo(name="emotion_detection_textcnn",
            version="1.3.0",
            summary="Baidu's open-source Emotion Detection Model(TextCNN).",
            author="baidu-nlp",
            author_email="",
            type="nlp/sentiment_analysis")
class EmotionDetectionTextCNN(hub.NLPPredictionModule):

    def _initialize(self):
        """
        initialize with the necessary elements
        """
        self.pretrained_model_path = os.path.join(self.directory, "assets", "infer_model")
        self.vocab_path = os.path.join(self.directory, "assets", "vocab.txt")
        self.vocab = load_vocab(self.vocab_path)
        self._word_seg_module = None

        self.predict = self.emotion_classify

        self._set_config()

    @property
    def word_seg_module(self):
        """
        lac module
        """
        if not self._word_seg_module:
            self._word_seg_module = hub.Module(name="lac")
        return self._word_seg_module

    @serving
    def emotion_classify(self, texts=[], data={}, use_gpu=False, batch_size=1):
        """
        Get the emotion prediction results results with the texts as input
        Args:
             texts(list): the input texts to be predicted, if texts not data
             data(dict): key must be 'text', value is the texts to be predicted, if data not texts
             use_gpu(bool): whether use gpu to predict or not
             batch_size(int): the program deals once with one batch
        Returns:
             results(list): the emotion prediction results
        """
        if use_gpu:
            try:
                _places = os.environ["CUDA_VISIBLE_DEVICES"]
                int(_places[0])
            except:
                raise RuntimeError(
                    "Environment Variable CUDA_VISIBLE_DEVICES is not set correctly. If you wanna use gpu, please set CUDA_VISIBLE_DEVICES as cuda_device_id."
                )

        if texts != [] and isinstance(texts, list) and data == {}:
            predicted_data = texts
        elif texts == [] and isinstance(data, dict) and isinstance(data.get('text', None), list) and data['text']:
            predicted_data = data["text"]
        else:
            raise ValueError("The input data is inconsistent with expectations.")

        predicted_data = self.to_unicode(predicted_data)

        start_idx = 0
        iteration = int(math.ceil(len(predicted_data) / batch_size))
        results = []
        for i in range(iteration):
            if i < (iteration - 1):
                batch_data = predicted_data[start_idx:(start_idx + batch_size)]
            else:
                batch_data = predicted_data[start_idx:]
            start_idx = start_idx + batch_size
            processed_results = preprocess(self.word_seg_module, batch_data, self.vocab, use_gpu, batch_size)
            tensor_words = self.texts2tensor(processed_results)

            if use_gpu:
                batch_out = self.gpu_predictor.run([tensor_words])
            else:
                batch_out = self.cpu_predictor.run([tensor_words])
            batch_result = postprocess(batch_out[0], processed_results)
            results += batch_result
        return results

    def get_labels(self):
        """
        Get the labels which was used when pretraining
        Returns:
             self.labels(dict)
        """
        self.labels = {"positive": 2, "negative": 0, "neutral": 1}
        return self.labels

import math
import os

import paddlehub as hub
from .processor import load_vocab
from .processor import postprocess
from .processor import preprocess
from paddlehub.compat.task import tokenization
from paddlehub.module.module import moduleinfo
from paddlehub.module.module import serving


@moduleinfo(name="porn_detection_lstm",
            version="1.2.0",
            summary="Baidu's open-source Porn Detection Model.",
            author="baidu-nlp",
            author_email="",
            type="nlp/sentiment_analysis")
class PornDetectionLSTM(hub.NLPPredictionModule):

    def __init__(self):
        """
        initialize with the necessary elements
        """
        self.pretrained_model_path = os.path.join(self.directory, "infer_model")
        self.tokenizer_vocab_path = os.path.join(self.directory, "assets", "vocab.txt")
        self.vocab_path = os.path.join(self.directory, "assets", "word_dict.txt")
        self.vocab = load_vocab(self.vocab_path)
        self.sequence_max_len = 256
        self.tokenizer = tokenization.FullTokenizer(self.tokenizer_vocab_path)

        self.param_file = os.path.join(self.directory, "assets", "params.txt")

        self.predict = self.detection

        self._set_config()

    @serving
    def detection(self, texts=[], data={}, use_gpu=False, batch_size=1):
        """
        Get the porn prediction results results with the texts as input

        Args:
             texts(list): the input texts to be predicted, if texts not data
             data(dict): key must be 'text', value is the texts to be predicted, if data not texts
             use_gpu(bool): whether use gpu to predict or not
             batch_size(int): the program deals once with one batch

        Returns:
             results(list): the porn prediction results
        """
        try:
            _places = os.environ["CUDA_VISIBLE_DEVICES"]
            int(_places[0])
        except:
            use_gpu = False

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
            processed_results = preprocess(batch_data, self.tokenizer, self.vocab, self.sequence_max_len)
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
        self.labels = {"porn": 1, "not_porn": 0}
        return self.labels

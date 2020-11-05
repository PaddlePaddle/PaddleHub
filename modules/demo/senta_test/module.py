import argparse
import os

import paddlehub as hub
from paddlehub.module.module import runnable, moduleinfo

from senta_test.processor import load_vocab


@moduleinfo(
    name="senta_test",
    version="1.0.0",
    summary="This is a PaddleHub Module. Just for test.",
    author="anonymous",
    author_email="",
    type="nlp/sentiment_analysis",
)
class SentaTest:
    def __init__(self):
        # add arg parser
        self.parser = argparse.ArgumentParser(
            description="Run the senta_test module.", prog='hub run senta_test', usage='%(prog)s', add_help=True)
        self.parser.add_argument('--input_text', type=str, default=None, help="text to predict")

        # load word dict
        vocab_path = os.path.join(self.directory, "vocab.list")
        self.vocab = load_vocab(vocab_path)

    def sentiment_classify(self, texts):
        results = []
        for text in texts:
            sentiment = "positive"
            for word in self.vocab:
                if word in text:
                    sentiment = "negative"
                    break
            results.append({"text": text, "sentiment": sentiment})

        return results

    @runnable
    def run_cmd(self, argvs):
        args = self.parser.parse_args(argvs)
        texts = [args.input_text]
        return self.sentiment_classify(texts)

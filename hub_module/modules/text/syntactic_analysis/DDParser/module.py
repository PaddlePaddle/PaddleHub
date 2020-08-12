# -*- coding:utf-8 -*-
import os
import argparse

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from paddle import fluid
import paddlehub as hub
from paddlehub.module.module import serving, moduleinfo, runnable

try:
    from ddparser import DDParser as DDParserModel
except:
    raise ImportError(
        "The module requires additional dependencies: ddparser. Please run 'pip install ddparser' to install it."
    )


@moduleinfo(
    name="ddparser",
    version="1.0.0",
    summary="Baidu's open-source DDParser model.",
    author="baidu-nlp",
    author_email="",
    type="nlp/syntactic_analysis")
class ddparser(hub.NLPPredictionModule):
    def _initialize(self):
        """
        initialize with the necessary elements
        """
        self.ddp = DDParserModel(prob=True, use_pos=True)
        self.font = font_manager.FontProperties(
            fname=os.path.join(self.directory, "SourceHanSans-Regular.ttf"))

    @serving
    def serving_parse(self, texts=[], return_visual=False):
        results = self.parse(texts, return_visual)
        if return_visual:
            for i, result in enumerate(results):
                result['visual'] = result['visual'].tolist()

        return results

    def parse(self, texts=[], return_visual=False):
        """
        parse the dependency.

        Args:
            texts(list[list[str] or list[list[str]]]): the input texts to be parse. It should be a list with elements: untokenized string or tokens list.
            return_visual(bool): if set True, the result will contain the dependency visualization.

        Returns:
            results(list[dict]): a list, with elements corresponding to each of the elements in texts. The element is a dictionary of shape:
                {
                    'word': list[str], the tokenized words.
                    'head': list[int], the head ids.
                    'deprel': list[str], the dependency relation.
                    'prob': list[float], the prediction probility of the dependency relation.
                    'postag': list[str], the POS tag. If the element of the texts is list, the key 'postag' will not return.
                    'visual' : list[numpy.array]: the dependency visualization. Use cv2.imshow to show or cv2.imwrite to save it. If return_visual=False, it will not return.
                }
       """

        if not texts:
            return
        if all([isinstance(i, str) and i for i in texts]):
            do_parse = self.ddp.parse
        elif all([isinstance(i, list) and i for i in texts]):
            do_parse = self.ddp.parse_seg
        else:
            raise ValueError("All of the elements should be string or list")
        results = do_parse(texts)
        if return_visual:
            for result in results:
                result['visual'] = self.visualize(
                    result['word'], result['head'], result['deprel'])
        return results

    @runnable
    def run_cmd(self, argvs):
        """
        Run as a command
        """
        self.parser = argparse.ArgumentParser(
            description='Run the %s module.' % self.name,
            prog='hub run %s' % self.name,
            usage='%(prog)s',
            add_help=True)

        self.arg_input_group = self.parser.add_argument_group(
            title="Input options", description="Input data. Required")

        self.add_module_input_arg()

        args = self.parser.parse_args(argvs)

        input_data = self.check_input_data(args)

        results = self.parse(texts=input_data)

        return results

    def visualize(self, word, head, deprel):
        """
        Visualize the dependency.

        Args:
            word: list[str], the tokenized words.
            head: list[int], the head ids.
            deprel: list[str], the dependency relation.

        Returns:
            data: a numpy array, use cv2.imshow to show it or cv2.imwrite to save it.
        """
        nodes = ['ROOT'] + word
        x = list(range(len(nodes)))
        y = [0] * (len(nodes))
        fig, ax = plt.subplots()
        # control the picture size
        max_span = max([abs(i + 1 - j) for i, j in enumerate(head)])
        fig.set_size_inches((len(nodes), max_span / 2))
        # set the points
        plt.scatter(x, y, c='w')

        for i in range(len(nodes)):
            txt = nodes[i]
            xytext = (i, 0)
            if i == 0:
                # set 'ROOT'
                ax.annotate(
                    txt,
                    xy=xytext,
                    xycoords='data',
                    xytext=xytext,
                    textcoords='data',
                )
            else:
                xy = (head[i - 1], 0)
                rad = 0.5 if head[i - 1] < i else -0.5
                # set the word
                ax.annotate(
                    txt,
                    xy=xy,
                    xycoords='data',
                    xytext=(xytext[0] - 0.1, xytext[1]),
                    textcoords='data',
                    fontproperties=self.font)
                # draw the curve
                ax.annotate(
                    "",
                    xy=xy,
                    xycoords='data',
                    xytext=xytext,
                    textcoords='data',
                    arrowprops=dict(
                        arrowstyle="<-",
                        shrinkA=12,
                        shrinkB=12,
                        color='blue',
                        connectionstyle="arc3,rad=%s" % rad,
                    ),
                )
                # set the deprel label. Calculate its position by the radius
                text_x = min(i, head[i - 1]) + abs((i - head[i - 1])) / 2 - 0.2
                text_y = abs((i - head[i - 1])) / 4
                ax.annotate(
                    deprel[i - 1],
                    xy=xy,
                    xycoords='data',
                    xytext=[text_x, text_y],
                    textcoords='data')

        # control the axis
        plt.axis('equal')
        plt.axis('off')

        # save to numpy array
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] +
                            (3, ))[:, :, ::-1]
        return data


if __name__ == "__main__":
    module = ddparser()
    # Data to be predicted
    test_text = ["百度是一家高科技公司"]
    results = module.parse(texts=test_text)
    print(results)
    test_tokens = [['百度', '是', '一家', '高科技', '公司']]
    results = module.parse(texts=test_text, return_visual=True)
    print(results)
    result = results[0]
    data = module.visualize(result['word'], result['head'], result['deprel'])
    import cv2
    import numpy as np
    cv2.imwrite('test1.jpg', data)
    cv2.imwrite('test2.jpg', result['visual'])

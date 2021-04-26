======================
How to custom Module
======================


I. Preparation
=======================

Basic Model Information
------------------------

We are going to write a PaddleHub Module with the following basic information about the module:

.. code-block:: yaml

    name: senta_test
    version: 1.0.0
    summary: This is a PaddleHub Module. Just for test.
    author: anonymous
    author_email:
    type: nlp/sentiment_analysis

The Module has an interface sentiment_classify, which is used to receive incoming text and give it a sentiment preference (positive/negative). It supports python interface calls and command line calls.

.. code-block:: python

    import paddlehub as hub

    senta_test = hub.Module(name="senta_test")
    senta_test.sentiment_classify(texts=["这部电影太差劲了"])

.. code-block:: shell

    hub run senta_test --input_text 这部电影太差劲了

Strategy
------------------------

For the sake of simplicity of the sample codes, we use a very simple sentiment strategy. When the input text has the word specified in the vocabulary list, the text tendency is judged to be negative; otherwise it is positive.

II. Create Module
=======================

Step 1: Create the necessary directories and files.
----------------------------------------------------

Create a senta_test directory. Then, create module.py, processor.py, and vocab.list in the senta_test directory, respectively.

.. code-block:: shell

    $ tree senta_test
    senta_test/
    ├── vocab.list 
    ├── module.py 
    └── processor.py 

============    =========================================================================
File Name       Purpose                                                      
------------    -------------------------------------------------------------------------
============    =========================================================================
module.py       It is the main module that provides the implementation codes of Module. 
processor.py    It is the helper module that provides a way to load the vocabulary list.
vocab.list      It stores the vocabulary. 
============    =========================================================================


Step 2: Implement the helper module processor.
------------------------------------------------

Implement a load_vocab interface in processor.py to read the vocabulary  list.

.. code-block:: python

    def load_vocab(vocab_path):
        with open(vocab_path) as file:
            return file.read().split()

Step 3: Write Module processing codes.
------------------------------------------------

The module.py file is the place where the Module entry code is located. We need to implement prediction logic on it.

Step 3_1. Reference the necessary header files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import argparse
    import os

    import paddlehub as hub
    from paddlehub.module.module import runnable, moduleinfo

    from senta_test.processor import load_vocab

.. note::

    When referencing a module in Module, you need to enter the full path, for example, senta_test. processor.

Step 3_2. Define the SentaTest class.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Module.py needs to have a class that inherits hub. Module, and this class is responsible for implementing the prediction logic and filling in basic information with using moduleinfo. When the hub. Module(name="senta\_test") is used to load Module, PaddleHub automatically creates an object of SentaTest and return it.

.. code-block:: python

    @moduleinfo(
        name="senta_test",
        version="1.0.0",
        summary="This is a PaddleHub Module. Just for test.",
        author="anonymous",
        author_email="",
        type="nlp/sentiment_analysis",
    )
    class SentaTest:
        ...

Step 3_3. Perform necessary initialization.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

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
                description="Run the senta_test module.",
                prog='hub run senta_test',
                usage='%(prog)s',
                add_help=True)
            self.parser.add_argument(
                '--input_text', type=str, default=None, help="text to predict")

            # load word dict
            vocab_path = os.path.join(self.directory, "vocab.list")
            self.vocab = load_vocab(vocab_path)

        ...

.. note::

    The execution class object has a built-in directory attribute by default. You can directly get the path of the Module.

Step 3_4: Refine the prediction logic.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    @moduleinfo(
        name="senta_test",
        version="1.0.0",
        summary="This is a PaddleHub Module. Just for test.",
        author="anonymous",
        author_email="",
        type="nlp/sentiment_analysis",
    )
    class SentaTest:
        ...

        def sentiment_classify(self, texts):
            results = []
            for text in texts:
                sentiment = "positive"
                for word in self.vocab:
                    if word in text:
                        sentiment = "negative"
                        break
                results.append({"text":text, "sentiment":sentiment})

            return results
        
        ...

Step 3_5. Support the command-line invoke.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want the module to support command-line invoke, you need to provide a runnable modified interface that parses the incoming data, makes prediction, and returns the results.

If you don't want to provide command-line prediction, you can leave the interface alone and PaddleHub automatically finds out that the module does not support command-line methods and gives a hint when PaddleHub executes in command lines.

.. code-block:: python

    @moduleinfo(
        name="senta_test",
        version="1.0.0",
        summary="This is a PaddleHub Module. Just for test.",
        author="anonymous",
        author_email="",
        type="nlp/sentiment_analysis",
    )
    class SentaTest:
        ...

        @runnable
        def run_cmd(self, argvs):
            args = self.parser.parse_args(argvs)
            texts = [args.input_text]
            return self.sentiment_classify(texts)

        ...

step 3_6. Support the serving invoke.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want the module to support the PaddleHub Serving deployment prediction service, you need to provide a serving-modified interface that parses the incoming data, makes prediction, and returns the results.

If you do not want to provide the PaddleHub Serving deployment prediction service, you do not need to add the serving modification.

.. code-block:: python

    @moduleinfo(
        name="senta_test",
        version="1.0.0",
        summary="This is a PaddleHub Module. Just for test.",
        author="anonymous",
        author_email="",
        type="nlp/sentiment_analysis",
    )
    class SentaTest:
        ...

        @serving
        def sentiment_classify(self, texts):
            results = []
            for text in texts:
                sentiment = "positive"
                for word in self.vocab:
                    if word in text:
                        sentiment = "negative"
                        break
                results.append({"text":text, "sentiment":sentiment})

            return results

Complete Code
------------------------------------------------

* `module.py <https://github.com/PaddlePaddle/PaddleHub/blob/release/v2.1/modules/demo/senta_test/module.py>`_

* `processor.py <https://github.com/PaddlePaddle/PaddleHub/blob/release/v2.1/modules/demo/senta_test/processor.py>`_

III. Install and test Module.
===================================

After writing a module, we can test it in the following ways:

Call Method 1
------------------------------------------------

Install the Module into the local machine, and then load it through Hub.Module(name=...)

.. code-block:: console

    $ hub install senta_test


.. code-block:: python

    import paddlehub as hub

    senta_test = hub.Module(name="senta_test")
    senta_test.sentiment_classify(texts=["这部电影太差劲了"])

Call Method 2
------------------------------------------------

Load directly through Hub.Module(directory=...)

.. code-block:: python

    import paddlehub as hub

    senta_test = hub.Module(directory="senta_test/")
    senta_test.sentiment_classify(texts=["这部电影太差劲了"])

Call Method 3
------------------------------------------------

Install the Module on the local machine and run it through hub run.

.. code-block:: console

    $ hub install senta_test
    $ hub run senta_test --input_text "这部电影太差劲了"
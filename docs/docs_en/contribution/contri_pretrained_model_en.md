# How to Write a PaddleHub Module and Go Live

## I. Preparation

### Basic Model Information

We are going to write a PaddleHub Module with the following basic information about the module:

```yaml
name: senta_test
version: 1.0.0
summary: This is a PaddleHub Module. Just for test.
author: anonymous
author_email:
type: nlp/sentiment_analysis
```

**This sample code can be referred to as [senta\_module\_sample](../../demo/senta_module_sample/senta_test)**

The Module has an interface sentiment\_classify, which is used to receive incoming text and give it a sentiment preference (positive/negative). It supports python interface calls and command line calls.

```python
import paddlehub as hub

senta_test = hub.Module(name="senta_test")
senta_test.sentiment_classify(texts=["这部电影太差劲了"])
```

```cmd
hub run senta_test --input_text 这部电影太差劲了
```

<br/>
### Strategy

For the sake of simplicity of the sample codes, we use a very simple sentiment strategy. When the input text has the word specified in the vocabulary list, the text tendency is judged to be negative; otherwise it is positive.

<br/>
## II. Create Module

### Step 1: Create the necessary directories and files.

Create a senta\_test directory. Then, create module.py, processor.py, and vocab.list in the senta\_test directory, respectively.

| File Name    | Purpose                                                      |
| ------------ | ------------------------------------------------------------ |
| module.py    | It is the main module that provides the implementation codes of Module. |
| processor.py | It is the helper module that provides a way to load the vocabulary list. |
| vocab.list   | It stores the vocabulary.                                    |

```cmd
➜  tree senta_test
senta_test/
├── vocab.list
├── module.py
└── processor.py
```

### Step 2: Implement the helper module processor.

Implement a load\_vocab interface in processor.py to read the vocabulary  list.

```python
def load_vocab(vocab_path):
    with open(vocab_path) as file:
        return file.read().split()
```

### Step 3: Write Module processing codes.

The module.py file is the place where the Module entry code is located. We need to implement prediction logic on it.

#### Step 3\_1. Reference the necessary header files

```python
import argparse
import os

import paddlehub as hub
from paddlehub.module.module import runnable, moduleinfo

from senta_test.processor import load_vocab
```

**NOTE:** When referencing a module in Module, you need to enter the full path, for example, senta\_test. processor.

#### Step 3\_2. Define the SentaTest class.

Module.py needs to have a class that inherits hub. Module, and this class is responsible for implementing the prediction logic and filling in basic information with using moduleinfo. When the hub. Module(name="senta\_test") is used to load Module, PaddleHub automatically creates an object of SentaTest and return it.

```python
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
```

#### Step 3\_3. Perform necessary initialization.

```python
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
```

`注意`: The execution class object has a built-in directory attribute by default. You can directly get the path of the Module.

#### Step 3\_4: Refine the prediction logic.

```python
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
```

#### Step 3\_5. Support the command-line invoke.

If you want the module to support command-line invoke, you need to provide a runnable modified interface that parses the incoming data, makes prediction, and returns the results.

If you don't want to provide command-line prediction, you can leave the interface alone and PaddleHub automatically finds out that the module does not support command-line methods and gives a hint when PaddleHub executes in command lines.

```python
@runnable
def run_cmd(self, argvs):
    args = self.parser.parse_args(argvs)
    texts = [args.input_text]
    return self.sentiment_classify(texts)
```

#### step 3\_6. Support the serving invoke.

If you want the module to support the PaddleHub Serving deployment prediction service, you need to provide a serving-modified interface that parses the incoming data, makes prediction, and returns the results.

If you do not want to provide the PaddleHub Serving deployment prediction service, you do not need to add the serving modification.

```python
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
```

### Complete Code

* [module.py](../../../modules/demo/senta_test/module.py)

* [processor.py](../../../modules/demo/senta_test/processor.py)

<br/>
## III. Install and test Module.

After writing a module, we can test it in the following ways:

### Call Method 1

Install the Module into the local machine, and then load it through Hub.Module(name=...)

```shell
➜  hub install senta_test
```

```python
import paddlehub as hub

senta_test = hub.Module(name="senta_test")
senta_test.sentiment_classify(texts=["这部电影太差劲了"])
```

### Call Method 2

Load directly through Hub.Module(directory=...)

```python
import paddlehub as hub

senta_test = hub.Module(directory="senta_test/")
senta_test.sentiment_classify(texts=["这部电影太差劲了"])
```

### Call Method 3

Load SentaTest object directly by adding senta\_test as a path to the environment variable.

```shell
➜  export PYTHONPATH=senta_test:$PYTHONPATH
```

```python
from senta_test.module import SentaTest

SentaTest.sentiment_classify(texts=["这部电影太差劲了"])
```

### Call Method 4

Install the Module on the local machine and run it through hub run.

```shell
➜  hub install senta_test
➜  hub run senta_test --input_text "这部电影太差劲了"
```

## IV. Release Module

After completing the development and testing of the module, if you want to share the model with others, you can release the model in the following ways.

### Method 1: Upload the Module to the PaddleHub website.

https://www.paddlepaddle.org.cn/hub

We will complete the review of the module and give feedback in the shortest possible time. After passing the review and going online, the module will be displayed on the PaddleHub website, and users can load it like any other official modules.

### Method 2: Upload the Module to the remote code hosting platform.

PaddleHub also supports loading Modules directly to the remote code hosting platforms. The steps are as follows:

#### Step 1: Create a new repository.

To create a new Git repository on the code hosting platform, add the codes of the module we wrote earlier. To make it easier to manage different modules, we create a modules directory and put senta\_test in the modules directory.

#### Step 2: Add a new configuration file`hubconf.py`.

In the root directory, add a new configuration `hubconf.py` file, which references a class modified by `moduleinfo` as follows:

```python
from modules.senta_test.module import SentaTest
```

*The structure of the file at this point is as follows:*

```
hubconf.py
modules
├── senta_test/
    ├── vocab.list
    ├── module.py
    └── processor.py
```

#### Step 3: Complete the commit and push to the remote repository.

#### Step 4: Load Module in the remote repository locally.

To facilitate the experience, we have stored the SentaTest codes on GitHub and Gitee. So you can directly experience the effect in the following ways:

```python
import paddlehub as hub

senta_test = hub.Module(name='senta_test', source='https://github.com/nepeplwu/myhub.git')
# senta_test = hub.Module(name='senta_test', source='https://gitee.com/nepeplwu/myhub.git')
print(senta_test.sentiment_classify(texts=["这部电影太差劲了"]))
```

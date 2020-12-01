# PaddleHub Command Line Tool

PaddleHub provides the command line tool for the management and use of pre-training models.

PaddleHub supports the change of the path of storing the pre-training models.

* If `${HUB_HOME}`Environment Variable is set, the pre-training models and configuration files are stored in the path specified by `${HUB_HOME}`.
* If the `${HUB_HOME}`Environment variable is not set, these files are stored in the path specified by `$HOME`.

The following 11 commands are currently supported on the command line.

## `hub install`

Installs Module locally. By default, it is installed in the `${HUB_HOME}/.paddlehub/modules` directory. When a module is installed locally, users can operate the Module through other commands (e.g., use the Module for prediction), or use the python API provided by PaddleHub to apply the Module to their own tasks to achieve migration learning.

## `hub uninstall`

Uninstalls local Modules.

## `hub show`

Views properties of locally installed modules or properties of modules in a specified directory, including name, version, description, author and other information.

## `hub download`

Downloads the Module provided by PaddleHub.

## `hub search`

Search the matching Module on the server by keywords. When you want to find the Module of a specific model, you can run the search command to quickly get the result. For example, the `hub search ssd` command runs to search for all Modules containing the word ssd. The command supports regular expressions, for example, `hub search ^s.*` runs to search all resources beginning with s.

`Note` If you want to search all Modules, the running of `hub search *` does not work. Because the shell expands its own wildcard, replacing \* with the filename in the current directory. For global search, users can type `hub search` directly.

## `hub list`

Lists the locally installed Modules

## `hub run`

Executes the prediction of Module. It should be noted that not all models support prediction (and likewise, not all models support migration learning). For more details on the run command, see `About Prediction` below.

## `hub help`

Displays help information.

## `hub version`

Displays the PaddleHub version information.

## `hub clear`

PaddleHub generates some cached data in the operation, which is stored in ${HUB\_HOME}/.paddlehub/cache directory by default. Users can clear the cache by running the clear command.

## `hub config`

Views and configures paddlehub-related information, including server address and log level.

`Example`

* `hub config`: Displays the current paddlehub settings.

* `hub config reset`: Restores current paddlehub settings to default settings

* `hub config server==[address]`: Sets the current paddlehub-server address to \[address], and paddlehub client gets model information from this address.

* `hub config log.level==[level]`: Sets the current log level to \[level]. Options are CRITICAL, ERROR, WARNING, EVAL, TRAIN, INFO, DEBUG, from left to right, from high priority to low priority.

* `hub config log.enable==True|False`: Sets whether the current log is available.

## `hub serving`

Deploys Module prediction service in one key. For details, see PaddleHub Serving One-Key Service Deployment.

**NOTE:**

In PaddleHub, Module represents a executable pre-training model`可执行的预训练模型`. A Module can support direct command-line prediction, or with the PaddleHub Finetune API. It implements the migration learning through a small number of codes. Not all Modules support command line prediction (for example, for BERT/ERNIE Transformer model, finetune is performed generally with a task). Not all Modules can be used for fine-tune (for example, for the LAC lexical analysis model, we do not recommend users to use finetune).

PaddleHub tries to simplify the cost of understanding when using command line predictions. Generally, the predictions are classified into two categories: NLP and CV.

## NLP Class Tasks

You can input data specified by --input\_text or --input\_file. Take the Baidu LAC model (Chinese lexical analysis) as an example. You can use the following two commands to analyze the single-line texts and multi-line texts.

```shell
# Single-line of sentence
$ hub run lac --input_text "It is a nice day today！"
```

```shell
# multi-line of sentences
$ hub run lac --input_file test.txt
```

The sample format of test.txt is as follows (each line is a sentence requiring lexical analysis):

```
It is a nice day today！
It is going to rain outside.
……more……
```

## CV Class Tasks

Input data is specified by `--input_path` or `--input_file`. Take the SSD model (single-stage object detection) as an example. Single-image and multi-image predictions can be performed by running the following two commands:

```shell
# Single Picture
$ hub run resnet_v2_50_imagenet --input_path test.jpg
```

```shell
# List of Pictures
$ hub run resnet_v2_50_imagenet --input_file test.txt
```

The format of test.txt is

```
cat.jpg
dog.jpg
person.jpg
……more……
```

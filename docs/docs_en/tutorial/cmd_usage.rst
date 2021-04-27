===========================
PaddleHub Command Line Tool
===========================

PaddleHub provides the command line tool for the management and use of pre-training models. 

There are 11 commands in total, covering various aspects such as model installation, uninstallation, prediction, etc.


hub install
==================

Installs Module locally. By default, it is installed in the `${HUB_HOME}/.paddlehub/modules` directory. When a module is installed locally, users can operate the Module through other commands (e.g., use the Module for prediction), or use the python API provided by PaddleHub to apply the Module to their own tasks to achieve migration learning.

.. tip::

    If Environment Variable *${HUB_HOME}* is set, the pre-training models and configuration files are stored in the path specified by *${HUB_HOME}*.
    If Environment Variable *${HUB_HOME}* is not set, these files are stored in the path specified by *$HOME*.

hub uninstall
==================

Uninstalls local Modules.

hub show
==================

Views properties of locally installed modules or properties of modules in a specified directory, including name, version, description, author and other information.

hub download
==================

Downloads the Module provided by PaddleHub.

hub search
==================

Search the matching Module on the server by keywords. When you want to find the Module of a specific model, you can run the search command to quickly get the result. For example, the `hub search ssd` command runs to search for all Modules containing the word ssd. The command supports regular expressions, for example, `hub search ^s.*` runs to search all resources beginning with s.

.. tip::
    
    If you want to search all Modules, the running of `hub search *` does not work. Because the shell expands its own wildcard, replacing \* with the filename in the current directory. For global search, users can type `hub search` directly.

hub list
==================

Lists the locally installed Modules

hub run
==================

Executes the prediction of Module. It should be noted that not all models support prediction (and likewise, not all models support migration learning). For more details please refer to the [Quick experience of *hub run*](../quick_experience/cmd_quick_run_en.md)

PaddleHub tries to simplify the cost of understanding when using command line predictions. Generally, the predictions are classified into two categories: NLP and CV.

NLP Class Tasks
---------------

You can input data specified by *--input_text*. Take the Baidu LAC model (Chinese lexical analysis) as an example. You can use the following command to analyze texts.


.. code-block:: console

    $ hub run lac --input_text "今天是个好日子"

CV Class Tasks
---------------

Input data is specified by *--input\_path*. Take the SSD model (single-stage object detection) as an example. Predictions can be performed by running the following command:

.. code-block:: console

    $ hub run resnet_v2_50_imagenet --input_path test.jpg

.. note::

    In PaddleHub, Module represents a executable pre-training model. A Module can support direct command-line prediction, or with the PaddleHub Finetune API. It implements the migration learning through a small number of codes. Not all Modules support command line prediction (for example, for BERT/ERNIE Transformer model, finetune is performed generally with a task). Not all Modules can be used for fine-tune (for example, for the LAC lexical analysis model, we do not recommend users to use finetune).

hub help
==================

Displays help information.

hub version
==================

Displays the PaddleHub version information.

hub clear
==================

PaddleHub generates some cached data in the operation, which is stored in ${HUB\_HOME}/.paddlehub/cache directory by default. Users can clear the cache by running the clear command.

hub config
==================

Views and configures paddlehub-related information, including server address and log level.

.. code-block:: console

    $ # Displays the current paddlehub settings.
    $ hub config 
    
    $ # Restores current paddlehub settings to default settings.
    $ hub config reset 
    
    $ # Sets the current paddlehub-server address to ${HOST}, and paddlehub client gets model information from this address.
    $ hub config server==${HOST} 
    
    $ # Sets the current log level to ${LEVEL}. Options are CRITICAL, ERROR, WARNING, EVAL, TRAIN, INFO, DEBUG, from left to right, from high priority to low priority.
    $ hub config log.level==${LEVEL} 
    
    $ # Sets whether the current log is available.
    $ hub config log.enable==True|False 

hub serving
==================

Deploys Module prediction service in one key. For details, see `PaddleHub Serving Deployment <serving>`_.
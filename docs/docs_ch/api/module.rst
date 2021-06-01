==============
Module
==============

.. code-block:: python

    class paddlehub.Module(
        name: str = None,
        directory: str = None,
        version: str = None,
        ignore_env_mismatch: bool = False,
        **kwargs)

-----------------

   In PaddleHub, Module represents an executable module, which usually a pre-trained model that can be used for end-to-end prediction, such as a face detection model or a lexical analysis model, or a pre-trained model that requires finetuning, such as BERT/ERNIE. When loading a Module with a specified name, if the Module does not exist locally, PaddleHub will automatically request the server or the specified Git source to download the resource.

-----------------

* Args:
    * name(str | optional)
        Module name.

    * directory(str | optional)
        Directory of the module to be loaded, only takes effect when the `name` is not specified.

    * version(str | optional)
        The version limit of the module, only takes effect when the `name` is specified. When the local Module does not meet the specified version conditions, PaddleHub will re-request the server to download the appropriate Module. Default to None, This means that the local Module will be used. If the Module does not exist, PaddleHub will download the latest version available from the server according to the usage environment.
    
    * ignore_env_mismatch(bool | optional)
        Whether to ignore the environment mismatch when installing the Module. Default to False.

**member functions**
=====================

export_onnx_model
------------------

    .. code-block:: python

        def export_onnx_model(
            dirname: str,
            input_spec: List[paddle.static.InputSpec] = None,
            include_sub_modules: bool = True,
            **kwargs):

    Export the model to ONNX format.

    * Args:
        * dirname(str)
            The directory to save the onnx model.
        
        * input_spec(list)
            Describes the input of the saved model's forward method, which can be described by InputSpec or example Tensor. If None, all input variables of the original Layer's forward method would be the inputs of the saved model. Default None.
            
        * include_sub_modules(bool)
            Whether to export sub modules. Default to True.
            
        * \*\*kwargs(dict|optional)
            Other export configuration options for compatibility, some may be removed in the future. Don't use them If not necessary. Refer to https://github.com/PaddlePaddle/paddle2onnx for more information.

save_inference_model
----------------------

    .. code-block:: python

        def save_inference_model(
            dirname: str,
            model_filename: str = None,
            params_filename: str = None,
            input_spec: List[paddle.static.InputSpec] = None,
            include_sub_modules: bool = True,
            combined: bool = True):

    Export the model to Paddle Inference format.

    * Args:
        * name(str | optional)
            Module name.

        * model_filename(str)
            The name of the saved model file. Default to `__model__`.

        * params_filename(str)
            The name of the saved parameters file, only takes effect when `combined` is True. Default to `__params__`.

        * input_spec(list)
            Describes the input of the saved model's forward method, which can be described by InputSpec or example Tensor. If None, all input variables of the original Layer's forward method would be the inputs of the saved model. Default None.

        * include_sub_modules(bool)
            Whether to export sub modules. Default to True.
        
        * combined(bool)
            Whether to save all parameters in a combined file. Default to True.

sub_modules
----------------------

    .. code-block:: python

        def sub_modules(recursive: bool = True):

    Get all sub modules.

    * Args:
        * recursive(bool): 
            Whether to get sub modules recursively. Default to True.

**classmethod**
=================

get_py_requirements
----------------------

    .. code-block:: python

        @classmethod
        def get_py_requirements(cls) -> List[str]:

    Get Module's python package dependency list.

load
----------------------

    .. code-block:: python

        @classmethod
        def load(cls, directory: str) -> Generic:

    Load the Module object defined in the specified directory.

    * Args:
        * directory(str): 
            Module directory.

load_module_info
----------------------

    .. code-block:: python

        @classmethod
        def load_module_info(cls, directory: str) -> EasyDict:

    Load the Module object defined in the specified directory.

    * Args:
        * directory(str): 
            Module directory.

**property**
=================

is_runnable
-----------------

    .. code-block:: python

        is_runnable

    Whether the Module is runnable, in other words, whether can we execute the Module through the `hub run` command.

name
-----------------

    .. code-block:: python

        name

    Module name.

directory
-----------------

    .. code-block:: python

        directory

    Directory of Module.

version
-----------------

    .. code-block:: python

        version

    Module name.

type
-----------------

    .. code-block:: python

        type

    Module type.

summary
-----------------

    .. code-block:: python

        summary

    Module summary.

author
-----------------

    .. code-block:: python

        author

    The author of Module

author_email
-----------------

    .. code-block:: python

        author_email

    The email of Module author

.. note::
    Module is a factory class that is used to automatically download and load user-defined model classes. In addition to the above methods or property, each Module has other custom methods or property. The relevant definitions need to be viewed in the corresponding documentation

# Class `hub.Module`

```python
hub.Module(
    name: str = None,
    directory: str = None,
    version: str = None,
    source: str = None,
    update: bool = False,
    branch: str = None,
    **kwargs):)
```

In PaddleHub, Module represents an executable module, which usually a pre-trained model that can be used for end-to-end prediction, such as a face detection model or a lexical analysis model, or a pre-trained model that requires finetuning, such as BERT/ERNIE. When loading a Module with a specified name, if the Module does not exist locally, PaddleHub will automatically request the server or the specified Git source to download the resource.

**Args**
* name(str): Module name.
* directory(str|optional): Directory of the module to be loaded, only takes effect when the `name` is not specified.
* version(str|optional): The version limit of the module, only takes effect when the `name` is specified. When the local Module does not meet the specified version conditions, PaddleHub will re-request the server to download the appropriate Module. Default to None, This means that the local Module will be used. If the Module does not exist, PaddleHub will download the latest version available from the server according to the usage environment.
* source(str|optional): Url of a git repository. If this parameter is specified, PaddleHub will no longer download the specified Module from the default server, but will look for it in the specified repository. Default to None.
* update(bool|optional): Whether to update the locally cached git repository, only takes effect when the `source` is specified. Default to False.
* branch(str|optional): The branch of the specified git repository. Default to None.

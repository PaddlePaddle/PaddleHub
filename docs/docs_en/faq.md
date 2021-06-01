# FAQ

## Failed to install paddlehub via pip
`Could not find a version that satisfies the requirement paddlehub (from versions: )`

This may be because pip points to a pypi mirror source, which is not synchronized with the paddlehub version in time.

Try the following command:

```shell
$ pip install -i https://pypi.org/simple/ paddlehub
```

## When using paddlehub, raise an Exception like
`ModuleNotFoundError: No module named 'paddle'`

This is because PaddleHub depends on PaddlePaddle, and users need to install the appropriate PaddlePaddle version by themselves.

If the machine does not support GPU, use the following command to install the CPU version of PaddlePaddle:

```shell
$ pip install paddlepaddle
```

Or, use the following command to install the GPU version of PaddlePaddle:

```shell
$ pip install paddlepaddle-gpu
```

## Cannot download preset datasets and Modules.

You can use server_check() to check the connection status between the local and remote PaddleHub-Server

```python
import paddlehub
paddlehub.server_check()
# If the remote PaddleHub-Server can be connected, it will display Request Hub-Server successfully.
# Otherwise, it will display Request Hub-Server unsuccessfully.
```

## Does PaddleHub Module support multi-threading to speed up prediction?

Due to the limitations of PaddlePaddle itself, PaddleHub cannot speed up model prediction through multi-threaded concurrent execution.

## How to modify the default storage path of PaddleHub's Module?

Set environment variable ${HUB_HOME}.

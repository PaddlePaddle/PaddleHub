# Installation

## Environment Dependence

* Python>=3.6
* PaddlePaddle>=2.0.0rc
* Operating System: Windows/Mac/Linux

## Installation Command

Before installing the PaddleHub, install the PaddlePaddle deep learning framework first. For more installation instructions, refer to [PaddleQuickInstall](https://github.com/PaddlePaddle/PaddleHub)

```shell
pip install paddlehub==2.0.0b
```

In addition to the above dependences, PaddleHub's pre-training models and pre-set datasets need to be downloaded through connecting to the server. Make sure that the computer can access the network. You can run PaddleHub offline if the relevant datasets and pre-set models are already available locally.

**NOTE:** Make sure that the computer can access the external network when the PaddleHub is used to download datasets and pre-training models. You can use `server_check()` to check the connection status between the local and remote PaddleHub-Server in the following methods:

```python
import paddlehub
paddlehub.server_check()
# If OK, reply "Request Hub-Server successfully".
# If not OK, reply "Hub-Server unsuccessfully".
```

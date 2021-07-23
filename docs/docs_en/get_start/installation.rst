============
Installation
============


Environment Dependency
========================

* Operating System: Windows/Mac/Linux
* Python >= 3.6.2
* PaddlePaddle >= 2.0.0

.. code-block:: shell

    # Install gpu version of paddlepaddle
    pip install paddlepaddle-gpu -U

    # Or install cpu version of paddlepaddle
    # pip install paddlepaddle -U

Installation Command
========================

Before installing the PaddleHub, install the PaddlePaddle deep learning framework first. For more installation instructions, refer to `PaddleQuickInstall <https://www.paddlepaddle.org.cn/install/quick>`_.

.. code-block:: shell

    pip install paddlehub==2.1.0

In addition to the above dependences, PaddleHub's pre-training models and pre-set datasets need to be downloaded through connecting to the server. Make sure that the computer can access the network. You can run PaddleHub offline if the relevant datasets and pre-set models are already available locally.

.. note::
    Make sure that the computer can access the external network when the PaddleHub is used to download datasets and pre-training models. You can use `server_check()` to check the connection status between the local and remote PaddleHub-Server in the following methods:

.. code-block:: Python

    import paddlehub
    paddlehub.server_check()
    # If OK, reply "Request Hub-Server successfully".
    # If not OK, reply "Hub-Server unsuccessfully".
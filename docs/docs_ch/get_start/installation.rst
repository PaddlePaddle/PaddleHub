============
安装
============


环境依赖
========================

* 操作系统：Windows/Mac/Linux
* Python >= 3.6.2
* PaddlePaddle >= 2.0.0

.. code-block:: shell

    # 安装gpu版本的PaddlePaddle
    pip install paddlepaddle-gpu -U

    # 或者安装cpu版本的paddlepaddle
    # pip install paddlepaddle -U

安装命令
========================

在安装PaddleHub之前，请先安装PaddlePaddle深度学习框架，更多安装说明请查阅`飞桨快速安装 <https://www.paddlepaddle.org.cn/install/quick>`

.. code-block:: shell

    pip install paddlehub==2.1.0

除上述依赖外，PaddleHub的预训练模型和预置数据集需要连接服务端进行下载，请确保机器可以正常访问网络。若本地已存在相关的数据集和预训练模型，则可以离线运行PaddleHub。

.. note::

    使用PaddleHub下载数据集、预训练模型等，要求机器可以访问外网。可以使用`server_check()`可以检查本地与远端PaddleHub-Server的连接状态，使用方法如下：

.. code-block:: Python

    import paddlehub
    paddlehub.server_check()
    # 如果可以连接远端PaddleHub-Server，则显示Request Hub-Server successfully。
    # 如果无法连接远端PaddleHub-Server，则显示Request Hub-Server unsuccessfully。

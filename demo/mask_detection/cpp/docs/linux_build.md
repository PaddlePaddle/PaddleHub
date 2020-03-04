# Linux平台口罩人脸检测及分类模型C++预测部署

## 1. 系统和软件依赖

### 1.1 操作系统及硬件要求

- Ubuntu 14.04 或者 16.04 (其它平台未测试)
- GCC版本4.8.5 ~ 4.9.2
- 支持Intel MKL-DNN的CPU
- NOTE: 如需在Nvidia GPU运行，请自行安装CUDA 9.0 / 10.0 + CUDNN 7.3+ (不支持9.1/10.1版本的CUDA)

### 1.2 下载PaddlePaddle C++预测库

PaddlePaddle C++ 预测库主要分为CPU版本和GPU版本。

其中，GPU 版本支持`CUDA 10.0` 和 `CUDA 9.0`:

以下为各版本C++预测库的下载链接：

|  版本   | 链接  |
|  ----  | ----  |
| CPU+MKL版  | [fluid_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/1.6.3-cpu-avx-mkl/fluid_inference.tgz) |
| CUDA9.0+MKL 版  | [fluid_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/1.6.3-gpu-cuda9-cudnn7-avx-mkl/fluid_inference.tgz) |
| CUDA10.0+MKL 版 | [fluid_inference.tgz](https://paddle-inference-lib.bj.bcebos.com/1.6.3-gpu-cuda10-cudnn7-avx-mkl/fluid_inference.tgz) |

更多可用预测库版本，请点击以下链接下载:[C++预测库下载列表](https://paddlepaddle.org.cn/documentation/docs/zh/advanced_usage/deploy/inference/build_and_install_lib_cn.html)


下载并解压, 解压后的 `fluid_inference`目录包含的内容：
```
fluid_inference
├── paddle # paddle核心库和头文件
|
├── third_party # 第三方依赖库和头文件
|
└── version.txt # 版本和编译信息
```

**注意:** 请把解压后的目录放到合适的路径，**该目录路径后续会作为编译依赖**使用。

### 1.2 编译安装 OpenCV

```shell
# 1. 下载OpenCV3.4.6版本源代码
wget -c https://paddleseg.bj.bcebos.com/inference/opencv-3.4.6.zip
# 2. 解压
unzip opencv-3.4.6.zip && cd opencv-3.4.6
# 3. 创建build目录并编译, 这里安装到/root/projects/opencv3目录
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/opencv3 -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DWITH_IPP=OFF -DBUILD_IPP_IW=OFF -DWITH_LAPACK=OFF -DWITH_EIGEN=OFF -DCMAKE_INSTALL_LIBDIR=lib64 -DWITH_ZLIB=ON -DBUILD_ZLIB=ON -DWITH_JPEG=ON -DBUILD_JPEG=ON -DWITH_PNG=ON -DBUILD_PNG=ON -DWITH_TIFF=ON -DBUILD_TIFF=ON
make -j4
make install
```

其中参数 `CMAKE_INSTALL_PREFIX` 参数指定了安装路径, 上述操作完成后，`opencv` 被安装在 `$HOME/opencv3` 目录(用户也可选择其他路径)，**该目录后续作为编译依赖**。

## 2. 编译与运行

### 2.1 配置编译脚本

cd `PaddleHub/deploy/demo/mask_detector/`

打开文件`linux_build.sh`, 看到以下内容:
```shell
# Paddle 预测库路径
PADDLE_DIR=/PATH/TO/fluid_inference/
# OpenCV 库路径
OPENCV_DIR=/PATH/TO/opencv3gcc4.8/
# 是否使用GPU
WITH_GPU=ON
# CUDA库路径, 仅 WITH_GPU=ON 时设置
CUDA_LIB=/PATH/TO/CUDA_LIB64/
# CUDNN库路径，仅 WITH_GPU=ON 且 CUDA_LIB有效时设置
CUDNN_LIB=/PATH/TO/CUDNN_LIB64/

cd build
cmake .. \
    -DWITH_GPU=${WITH_GPU} \
    -DPADDLE_DIR=${PADDLE_DIR} \
    -DCUDA_LIB=${CUDA_LIB} \
    -DCUDNN_LIB=${CUDNN_LIB} \
    -DOPENCV_DIR=${OPENCV_DIR} \
    -DWITH_STATIC_LIB=OFF
make -j4
```

把上述参数根据实际情况做修改后，运行脚本编译程序：
```shell
sh linux_build.sh
```

### 2.2. 运行和可视化

可执行文件有 **2** 个参数，第一个是前面导出的`inference_model`路径，第二个是需要预测的图片路径。

示例:
```shell
./build/main /PATH/TO/pyramidbox_lite_server_mask/ /PATH/TO/TEST_IMAGE
```

执行程序时会打印检测框的位置与口罩是否佩戴的结果，另外result.jpg文件为检测的可视化结果。

**预测结果示例:**

![output_image](https://paddlehub.bj.bcebos.com/deploy/result.jpg)

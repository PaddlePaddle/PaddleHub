
WITH_GPU=ON

PADDLE_DIR=/ssd3/chenzeyu01/PaddleMaskDetector/fluid_inference
CUDA_LIB=/home/work/cuda-10.1/lib64/
CUDNN_LIB=/home/work/cudnn/cudnn_v7.4/cuda/lib64/
OPENCV_DIR=/ssd3/chenzeyu01/PaddleMaskDetector/opencv3gcc4.8/

rm -rf build
mkdir -p build
cd build

cmake .. \
    -DWITH_GPU=${WITH_GPU} \
    -DPADDLE_DIR=${PADDLE_DIR} \
    -DCUDA_LIB=${CUDA_LIB} \
    -DCUDNN_LIB=${CUDNN_LIB} \
    -DOPENCV_DIR=${OPENCV_DIR} \
    -DWITH_STATIC_LIB=OFF
make clean
make -j12

FROM ubuntu:16.04

RUN echo "deb [trusted=true] http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial main restricted \n\
deb [trusted=true] http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-updates main restricted \n\
deb [trusted=true] http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial universe \n\
deb [trusted=true] http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-updates universe \n\
deb [trusted=true] http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial multiverse \n\
deb [trusted=true] http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-updates multiverse \n\
deb [trusted=true] http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-backports main restricted universe multiverse \n\
deb [trusted=true] http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-security main restricted \n\
deb [trusted=true] http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-security universe \n\
deb [trusted=true] http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-security multiverse" > /etc/apt/sources.list

RUN apt-get update && apt-get install -y inetutils-ping wget vim curl cmake git sox libsndfile1 libpng12-dev \
    libpng-dev swig libzip-dev openssl bc libflac* libgdk-pixbuf2.0-dev libpango1.0-dev libcairo2-dev \
    libgtk2.0-dev pkg-config zip unzip zlib1g-dev libreadline-dev libbz2-dev liblapack-dev libjpeg-turbo8-dev \
    sudo lrzsz libsqlite3-dev libx11-dev libsm6 apt-utils libopencv-dev libavcodec-dev libavformat-dev \
    libswscale-dev locales liblzma-dev python-lzma m4 libxext-dev strace libibverbs-dev libpcre3 libpcre3-dev \
    build-essential libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev xz-utils \
    libfreetype6-dev libxslt1-dev libxml2-dev libgeos-3.5.0 libgeos-dev && apt-get install -y --allow-downgrades \
    --allow-change-held-packages && DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata \
    && /bin/cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && dpkg-reconfigure -f noninteractive tzdata

RUN echo "set meta-flag on" >> /etc/inputrc && echo "set convert-meta off" >> /etc/inputrc && \
    locale-gen en_US.UTF-8 && /sbin/ldconfig -v && groupadd -g 10001 paddlehub && \
    useradd -m -s /bin/bash -N -u 10001 paddlehub -g paddlehub && chmod g+w /etc/passwd && \
    echo "paddlehub ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 LANGUAGE=en_US.UTF-8 TZ=Asia/Shanghai

# official download site: https://www.python.org/ftp/python/3.7.13/Python-3.7.13.tgz
RUN wget https://cdn.npmmirror.com/binaries/python/3.7.13/Python-3.7.13.tgz && tar xvf Python-3.7.13.tgz && \
    cd Python-3.7.13 && ./configure --prefix=/home/paddlehub/python3.7 && make -j8 && make install && \
    rm -rf ../Python-3.7.13 ../Python-3.7.13.tgz && chown -R paddlehub:paddlehub /home/paddlehub/python3.7

RUN cd /tmp && wget https://mirrors.sjtug.sjtu.edu.cn/gnu/gmp/gmp-6.1.0.tar.bz2 && tar xvf gmp-6.1.0.tar.bz2 && \
    cd gmp-6.1.0 && ./configure --prefix=/usr/local && make -j8 && make install && \
    rm -rf ../gmp-6.1.0.tar.bz2 ../gmp-6.1.0 && cd /tmp && \
    wget https://www.mpfr.org/mpfr-3.1.4/mpfr-3.1.4.tar.bz2 && tar xvf mpfr-3.1.4.tar.bz2 && cd mpfr-3.1.4 && \
    ./configure --prefix=/usr/local && make -j8 && make install && rm -rf ../mpfr-3.1.4.tar.bz2 ../mpfr-3.1.4 && \
    cd /tmp && wget https://mirrors.sjtug.sjtu.edu.cn/gnu/mpc/mpc-1.0.3.tar.gz && tar xvf mpc-1.0.3.tar.gz && \
    cd mpc-1.0.3 && ./configure --prefix=/usr/local && make -j8 && make install && \
    rm -rf ../mpc-1.0.3.tar.gz ../mpc-1.0.3 && cd /tmp && \
    wget http://www.mirrorservice.org/sites/sourceware.org/pub/gcc/infrastructure/isl-0.18.tar.bz2 && \
    tar xvf isl-0.18.tar.bz2 && cd isl-0.18 && ./configure --prefix=/usr/local && make -j8 && make install \
    && rm -rf ../isl-0.18.tar.bz2 ../isl-0.18 && cd /tmp && \
    wget http://mirrors.ustc.edu.cn/gnu/gcc/gcc-8.2.0/gcc-8.2.0.tar.gz --no-check-certificate && \
    tar xvf gcc-8.2.0.tar.gz && cd gcc-8.2.0 && unset LIBRARY_PATH && ./configure --prefix=/home/paddlehub/gcc82 \
    --enable-threads=posix --disable-checking --disable-multilib --enable-languages=c,c++ --with-gmp=/usr/local \
    --with-mpfr=/usr/local --with-mpc=/usr/local --with-isl=/usr/local && make -j8 && make install && \
    rm -rf ../gcc-8.2.0.tar.gz ../gcc-8.2.0 && chown -R paddlehub:paddlehub /home/paddlehub/gcc82

WORKDIR /home/paddlehub
USER paddlehub 
ENV PATH=/home/paddlehub/python3.7/bin:/home/paddlehub/gcc82/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.2/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}

RUN mkdir -p ~/.pip && echo "[global]" > ~/.pip/pip.conf && \
    echo "index-url=https://mirror.baidu.com/pypi/simple" >> ~/.pip/pip.conf && \
    echo "trusted-host=mirror.baidu.com" >> ~/.pip/pip.conf && \
    pip3 install --upgrade pip && pip3 install paddlepaddle paddlehub shapely pyclipper && \
    sudo cp -f /home/paddlehub/gcc82/lib64/libstdc++.so.6.0.25 /usr/lib/x86_64-linux-gnu/libstdc++.so.6 && \
    rm -rf ~/.cache/pip 

#RUN hub install <model_name>
CMD ['bash']

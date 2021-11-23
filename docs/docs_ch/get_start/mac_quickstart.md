# 零基础mac安装并实现图像风格迁移

## 第1步：安装Anaconda

- 说明：使用paddlepaddle需要先安装python环境，这里我们选择python集成环境Anaconda工具包
  - Anaconda是1个常用的python包管理程序
  - 安装完Anaconda后，可以安装python环境，以及numpy等所需的工具包环境
- Anaconda下载：
  - 地址：https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/?C=M&O=D
  - <img src="../../imgs/Install_Related/mac/anaconda_start.png" alt="anaconda download" width="800" align="center"/>
  - 选择最下方的`Anaconda3-2021.05-MacOSX-x86_64.pkg`下载
- 下载完成后，双击.pkg文件进入图形界面
  - 按默认设置即可，安装需要花费一段时间
- 建议安装vscode或pycharm等代码编辑器

## 第2步：打开终端并创建conda环境

- 打开终端

  - 同时按下command键和空格键，在聚焦搜索中输入"终端"，双击进入终端

- **将conda加入环境变量**

  - 加入环境变量是为了让系统能识别conda命令

  - 输入以下命令，在终端中打开`~/.bash_profile`：

    - ```shell
      vim ~/.bash_profile
      ```

  - 在`~/.bash_profile`中将conda添加为环境变量：

    - ```shell
      # 先按i进入编辑模式
      # 在第一行输入：
      export PATH="~/opt/anaconda3/bin:$PATH"
      # 若安装时自定义了安装位置，则将~/opt/anaconda3/bin改为自定义的安装目录下的bin文件夹
      ```

    - ```shell
      # 修改后的~/.bash_profile文件应如下（其中xxx为用户名）：
      export PATH="~/opt/anaconda3/bin:$PATH"
      # >>> conda initialize >>>
      # !! Contents within this block are managed by 'conda init' !!
      __conda_setup="$('/Users/xxx/opt/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
      if [ $? -eq 0 ]; then
          eval "$__conda_setup"
      else
          if [ -f "/Users/xxx/opt/anaconda3/etc/profile.d/conda.sh" ]; then
              . "/Users/xxx/opt/anaconda3/etc/profile.d/conda.sh"
          else
              export PATH="/Users/xxx/opt/anaconda3/bin:$PATH"
          fi
      fi
      unset __conda_setup
      # <<< conda initialize <<<
      ```

    - 修改完成后，先按`esc`键退出编辑模式，再输入`:wq!`并回车，以保存退出

  - 验证是否能识别conda命令：

    - 在终端中输入`source ~/.bash_profile`以更新环境变量
    - 再在终端输入`conda info --envs`，若能显示当前有base环境，则conda已加入环境变量

- 创建新的conda环境

  - ```shell
    # 在命令行输入以下命令，创建名为paddle_env的环境
    # 此处为加速下载，使用清华源
    conda create --name paddle_env python=3.8 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
    ```

  - 该命令会创建1个名为paddle_env、python版本为3.8的可执行环境，根据网络状态，需要花费一段时间

  - 之后命令行中会输出提示信息，输入y并回车继续安装

    - <img src="../../imgs/Install_Related/mac/conda_create.png" alt="conda_create" width="600" align="center"/>

- 激活刚创建的conda环境，在命令行中输入以下命令：

  - ```shell
    # 激活paddle_env环境
    conda activate paddle_env
    # 查看当前python的位置
    where python
    ```

  - <img src="../../imgs/Install_Related/mac/conda_activate.png" alt="conda_actviate" width="600" align="center"/>

  - 以上anaconda环境和python环境安装完毕

## 第3步：安装程序所需要库

- 使用pip命令在刚激活的环境中安装paddle：

  - ```shell
    # 在命令行中输入以下命令
    # 确认当前所用的pip是否是paddle_env环境下的pip
    where pip
    # 默认安装CPU版本，安装paddle时建议使用百度源
    pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
    ```

- 安装完paddle后，继续在paddle_env环境中安装paddlehub：

  - ```shell
    # 在命令行中输入以下命令
    pip install paddlehub -i https://mirror.baidu.com/pypi/simple
    ```

  - paddlehub的介绍文档：https://github.com/PaddlePaddle/PaddleHub/blob/release/v2.1/README_ch.md
  
  - 安装paddlehub时会自动安装其它依赖库，可能需要花费一段时间

## 第4步：安装paddlehub并下载模型

- 安装完paddlehub后，下载风格迁移模型：

  - ```shell
    # 在命令行中输入以下命令
    hub install stylepro_artistic==1.0.1
    ```

  - 模型的说明文档：[https://www.paddlepaddle.org.cn/hubsearch?filter=en_category&value=%7B%22scenes%22%3A%5B%22GANs%22%5D%7D](https://www.paddlepaddle.org.cn/hubsearch?filter=en_category&value={"scenes"%3A["GANs"]})

  - <img src="../../imgs/Install_Related/mac/hub_model_intro.png" alt="hub model intro" width="800" align="center"/>

## 第5步：准备风格迁移数据和代码

### 准备风格迁移数据

- 在桌面创建工作目录`style_transfer`

  - ```shell
    # 在终端中输入以下命令:
    cd ~/Desktop  # 进入桌面
    mkdir style_transfer  # 创建style_transfer文件夹
    cd style_transfer  # 进入style_transfer目录
    ```

- 分别放置待转换图片和风格图片：

  - 将待转换图片放置到桌面的`style_transfer/pic.jpg`
    - <img src="../../imgs/Install_Related/mac/pic.jpg" alt="pic.jpg" width="400" align="center"/>
  - 将风格图片放置到桌面的`style_transfer/fangao.jpg`
    - <img src="../../imgs/Install_Related/mac/fangao.jpg" alt="fangao.jpg" width="350" align="center"/>

### 代码

- 在`style_transfer`目录下创建代码文件`style_transfer.py`

- 在`style_transfer.py`中复制进如下代码：

  - ```python
    import paddlehub as hub
    import cv2
    
    # 待转换图片的相对地址
    picture = './pic.jpg'
    # 风格图片的相对地址
    style_image = './fangao.jpg'
    
    # 创建风格转移网络并加载参数
    stylepro_artistic = hub.Module(name="stylepro_artistic")
    
    # 读入图片并开始风格转换
    result = stylepro_artistic.style_transfer(
                        images=[{'content': cv2.imread(picture),
                                 'styles': [cv2.imread(style_image)]}],
                        visualization=True
    )
    ```

- 若没有vscode等代码编辑器，则可通过命令行方法：

  - ```shell
    pwd # 查看当前目录是否为style_transfer，若不是则输入：cd ~/Desktop/style_transfer
    touch style_transfer.py  # 创建空文件
    vim style_transfer.py  # 使用vim编辑器打开代码文件
    # 先输入i进入编辑模式
    # 将上面的代码拷贝进vim编辑器中
    # 按esc键退出编辑模式，再输入":wq"并回车，以保存并退出
    ```

- 运行代码：

  - 在命令行中，输入`python style_transfer.py`
  - 程序执行时，会创建新文件夹`transfer_result`，并将转换后的文件保存到该目录下
  - 输出的图片如下：
    - <img src="../../imgs/Install_Related/mac/output_img.png" alt="output image" width="600" align="center"/>

## 第6步：飞桨预训练模型探索之旅
- 恭喜你，到这里PaddleHub在mac环境下的安装和入门案例就全部完成了，快快开启你更多的深度学习模型探索之旅吧。[【更多模型探索，跳转飞桨官网】](https://www.paddlepaddle.org.cn/hublist)






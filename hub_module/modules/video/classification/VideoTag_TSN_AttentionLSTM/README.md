```shell
$ hub install VideoTag_TSN_AttentionLSTM==1.0.0
```
<p align="center">
<img src="http://bj.bcebos.com/ibox-thumbnail98/6dccffb080a6a32f8d0a44411368110b?authorization=bce-auth-v1%2Ffbe74140929444858491fbf2b6bc0935%2F2020-05-09T09%3A11%3A36Z%2F1800%2F%2F6c1b6b33cabacaa17584b52c68d2dc679f73a44a909cf9331871a7c9f12eb8c0" hspace='10'/> <br />
</p>
具体网络结构可参考论文[TSN](https://arxiv.org/abs/1608.00859)和AttentionLSTM(https://arxiv.org/abs/1503.08909)。

## 命令行预测示例
```shell
hub run VideoTag_TSN_AttentionLSTM --input_path 1.mp4 --use_gpu False
```
**参数**
示例文件下载：
* [1.mp4](https://paddlehub.bj.bcebos.com/model/video/video_classifcation/1.mp4)
* [2.mp4](https://paddlehub.bj.bcebos.com/model/video/video_classifcation/2.mp4)

## API
```python
def classification(paths,
                   use_gpu=False,
                   top_k=10,
                   save_dir=None)
```

用于视频分类预测

**参数**

* paths(list\[str\])：mp4文件路径
* use_gpu(bool)：是否使用GPU预测，默认为False
* top_k(int): 返回预测结果的前k个，默认为10
* save_dir(str): 预测结果保存路径，默认为None

**返回**

* results(list\[dict\]): result中的每个元素为对应输入的预测结果，预测单个mp4文件时仅有1个元素。每个预测结果为dict，包含mp4文件路径path及其分类概率。例：
```shell
[{'path': '1.mp4', '训练': 0.9771281480789185, '蹲': 0.9389840960502625, '杠铃': 0.8554490804672241, '健身房': 0.8479971885681152, '肌肉': 0.04704030603170395, '壶铃': 0.018318576738238335, '健身': 0.01663289964199066, '哑铃': 0.01484287716448307, '人体': 0.0097474: 0.00965932197868824}, {'path': '2.mp4', '舞蹈': 0.8504238724708557, '表演艺术': 0.04463545233011246, '教学': 0.042246297001838684, '达人秀': 0.04014882817864418, '健身': 0.017126718536019325, '体操': 0.01548301987349987, '剑': 0.01502374280244112, '教程': 0.01394: 0.011269202455878258, '广场': 0.009399269707500935}]
```

**代码示例**

```python
import paddlehub as hub

videotag = hub.Module(name="VideoTag_TSN_AttentionLSTM")

# execute predict and print the result
results = videotag.classification(paths=["1.mp4","2.mp4"], use_gpu=True, save_dir="predict_dir")
for result in results:
    print(result)
```

## 依赖

paddlepaddle >= 1.6.2

paddlehub >= 1.6.0

## 更新历史

* 1.0.0

  初始发布

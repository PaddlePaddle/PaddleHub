```shell
$ hub install videotag_tsn_lstm==1.0.0
```
<p align="center">
<img src="https://paddlehub.bj.bcebos.com/model/video/video_classifcation/VideoTag_TSN_AttentionLSTM.png" hspace='10'/> <br />
</p>
具体网络结构可参考论文[TSN](https://arxiv.org/abs/1608.00859)和[AttentionLSTM](https://arxiv.org/abs/1503.08909)。

## 命令行预测示例
```shell
hub run videotag_tsn_lstm --input_path 1.mp4 --use_gpu False
```
示例文件下载：
* [1.mp4](https://paddlehub.bj.bcebos.com/model/video/video_classifcation/1.mp4)
* [2.mp4](https://paddlehub.bj.bcebos.com/model/video/video_classifcation/2.mp4)

## API
```python
def classification(paths,
                   use_gpu=False,
                   threshold=0.5,
                   top_k=10)
```

用于视频分类预测

**参数**

* paths(list\[str\])：mp4文件路径
* use_gpu(bool)：是否使用GPU预测，默认为False
* threshold(float)：预测结果阈值，只有预测概率大于阈值的类别会被返回，默认为0.5
* top_k(int): 返回预测结果的前k个，默认为10

**返回**

* results(list\[dict\]): result中的每个元素为对应输入的预测结果，预测单个mp4文件时仅有1个元素。每个预测结果为dict，包含mp4文件路径path及其分类概率。例：
```shell
[{'path': '1.mp4', 'prediction': {'训练': 0.9771281480789185, '蹲': 0.9389840960502625, '杠铃': 0.8554490804672241, '健身房': 0.8479971885681152}}, {'path': '2.mp4', 'prediction': {'舞蹈': 0.8504238724708557}}]
```

**代码示例**

```python
import paddlehub as hub

videotag = hub.Module(name="videotag_tsn_lstm")

# execute predict and print the result
results = videotag.classification(paths=["1.mp4","2.mp4"], use_gpu=True)
for result in results:
    print(result)
```

## 依赖

paddlepaddle >= 1.6.2

paddlehub >= 1.6.0

## 更新历史

* 1.0.0

  初始发布

```shell
$ hub install videotag_tsn_lstm==1.0.0
```

## 命令行预测示例
```shell
hub run videotag_tsn_lstm --video_path 1.mp4 --use_gpu True
hub run videotag_tsn_lstm --filelist mp4_list.txt --save_dir predict_dir
```
预测单个mp4文件请指定--video_path，预测多个mp4文件请指定--filelist，若同时指定--video_path与--filelist则只有--filelist生效。
可选参数use_gpu默认为False，当指定为True时使用GPU进行预测；可选参数save_dir默认为None，当指定输出路径时，预测结果将保存至指定文件夹中。
示例文件下载：
* [1.mp4](https://paddlehub.bj.bcebos.com/model/video/video_classifcation/1.mp4)
* [2.mp4](https://paddlehub.bj.bcebos.com/model/video/video_classifcation/2.mp4)
* [mp4_list.txt](https://paddlehub.bj.bcebos.com/model/video/video_classifcation/mp4_list.txt)

## API
```python
def classification(video_path=None, filelist=None, use_gpu=False, save_dir=None)
```

用于视频分类预测

**参数**

* video_path(str)：单个mp4文件路径， 默认为None
* filelist(str): mp4文件列表路径，默认为None，当同时指定video_path与filelist时，仅filelist生效
* use_gpu(bool)：是否使用GPU预测，默认为False
* save_dir(str): 预测结果保存路径，默认为None

**返回**

* results(list\[dict\]): result中的每个元素为对应输入的预测结果，预测单个mp4文件时仅有1个元素。每个预测结果为dict，包含mp4文件路径path及其分类概率。例：
```shell
[{'path': '1.mp4', '训练': 0.9771281480789185, '蹲': 0.9389840960502625, '杠铃': 0.8554490804672241, '健身房': 0.8479971885681152, '肌肉': 0.04704030603170395, '壶铃': 0.018318576738238335, '健身': 0.01663289964199066, '哑铃': 0.01484287716448307, '人体': 0.0097474: 0.00965932197868824}, {'path': '2.mp4', '舞蹈': 0.8504238724708557, '表演艺术': 0.04463545233011246, '教学': 0.042246297001838684, '达人秀': 0.04014882817864418, '健身': 0.017126718536019325, '体操': 0.01548301987349987, '剑': 0.01502374280244112, '教程': 0.01394: 0.011269202455878258, '广场': 0.009399269707500935}]
```

**代码示例**

```python
import paddlehub as hub

videotag = hub.Module(name="videotag_tsn_lstm")

# execute predict and print the result
results = videotag.classification(filelist="mp4_list.txt", use_gpu=True, save_dir="predict_dir")
for result in results:
    print(result)
```

## 依赖

paddlepaddle >= 1.6.2

paddlehub >= 1.6.0

## 更新历史

* 1.0.0

  初始发布

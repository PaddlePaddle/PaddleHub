# Customized Data

In the training of a new task, it is a time-consuming process starting from zero, without producing the desired results probably. You can use the pre-training model provided by PaddleHub for fine-tune of a specific task. You just need to perform pre-processing of the customized data accordingly, and then input the pre-training model to get the corresponding results. Refer to the following for setting up the structure of the dataset.

## I. Image Classification Dataset

When migrating a classification task using custom data with PaddleHub, you need to slice the dataset into training set, validation set, and test set.

### Data Preparation

Three text files are needed to record the corresponding image paths and labels, plus a label file to record the name of the label.

```
├─data:
  ├─train_list.txt：
  ├─test_list.txt：
  ├─validate_list.txt：
  ├─label_list.txt：
  └─...
```

The format of the data list file for the training/validation/test set is as follows:

```
Path-1  label-1
Path-2  label-2
...
```

The format of label\_list.txt is:

```
Classification 1
Classification 2
...
```

Example: Take [Flower Dataset](../reference/dataset.md) as an example, train\_list.txt/test\_list.txt/validate\_list.txt:

```
roses/8050213579_48e1e7109f.jpg 0
sunflowers/45045003_30bbd0a142_m.jpg 3
daisy/3415180846_d7b5cced14_m.jpg 2
```

label\_list.txt reads as follows.

```
roses
tulips
daisy
sunflowers
dandelion
```

### Dataset Loading

For the preparation code of dataset, see [flowers.py](../../paddlehub/datasets/flowers.py). `hub.datasets.Flowers()` It automatically downloads the dataset from the network and unzip it into the `$HOME/.paddlehub/dataset` directory. Specific usage:

```python
from paddlehub.datasets import Flowers

flowers = Flowers(transforms)
flowers_validate = Flowers(transforms, mode='val')
```

* `transforms`: Data pre-processing mode.
* `mode`: Select data mode. Options are `train`, `test`, `val`. Default is `train`.

## II. Image Coloring Dataset

When migrating a coloring task using custom data with PaddleHub, you need to slice the dataset into training set and test set.

### Data Preparation

You need to divide the color images for coloring training and testing into training set data and test set data.

```
├─data:
  ├─train：
      |-folder1
      |-folder2
      |-……
      |-pic1
      |-pic2
      |-……

  ├─test：
    |-folder1
    |-folder2
    |-……
    |-pic1
    |-pic2
    |-……
  └─……
```

Example: PaddleHub provides users with a dataset for coloring `Canvas dataset. It consists of 1193 images in Monet style and 400 images in Van Gogh style. Take [Canvas Dataset](../reference/datasets.md) as an example, the contents of the train folder are as follows:

```
├─train：
      |-monet
          |-pic1
          |-pic2
          |-……  
      |-vango
          |-pic1
          |-pic2
          |-……
```

### Dataset Loading

For dataset preparation codes, refer to [canvas.py](../../paddlehub/datasets/canvas.py). `hub.datasets.Canvas()` It automatically downloads the dataset from the network and unzip it into the `$HOME/.paddlehub/dataset` directory. Specific usage:

```python
from paddlehub.datasets import Canvas

color_set = Canvas(transforms, mode='train')
```

* `transforms`: Data pre-processing mode.
* `mode`: Select data mode. Options are `train`, `test`. The default is `train`.

## III. Style Migration Dataset

When using custom data for style migration tasks with PaddleHub, you need to slice the dataset into training set and test set.

### Data Preparation

You need to split the color images for style migration into training set and test set data.

```
├─data:
  ├─train：
      |-folder1
      |-folder2
      |-...
      |-pic1
      |-pic2
      |-...

  ├─test：
    |-folder1
    |-folder2
    |-...
    |-pic1
    |-pic1
    |-...
  |- 21styles
    ｜-pic1
    ｜-pic1
  └─...
```

Example: PaddleHub provides users with data sets for style migration `MiniCOCO dataset`. Training set data and test set data is from COCO2014, in which there are 2001 images in the training set and 200 images in the test set. There are 21 images with different styles in `21styles` folder. Users can change the images of different styles as needed. Take [MiniCOCO Dataset](../reference/datasets.md) as an example. The content of train folder is as follows:

```
├─train：
      |-train
          |-pic1
          |-pic2
          |-……  
      |-test
          |-pic1
          |-pic2
          |-……
      |-21styles
          |-pic1
          |-pic2
          |-……
```

### Dataset Loading

For the preparation codes of the dataset, refer to [minicoco.py](../../paddlehub/datasets/minicoco.py). `hub.datasets.MiniCOCO()` It automatically downloads the dataset from the network and unzip it into the `$HOME/.paddlehub/dataset` directory. Specific usage:

```python
from paddlehub.datasets import MiniCOCO

ccolor_set = MiniCOCO(transforms, mode='train')
```

* `transforms`: Data pre-processing mode.
* `mode`: Select data mode. Options are `train`, `test`. The default is `train`.

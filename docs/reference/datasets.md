# Class `hub.datasets.Canvas`

```python
hub.datasets.Canvas(
    transforms: Callable,
    mode: str = 'train')
```

Dataset for colorization. It contains 1193 and 400 pictures for Monet and Vango paintings style, respectively. We collected data from https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/.

**Args**
* transforms(callmethod) : The method of preprocess images.
* mode(str): The mode for preparing dataset.

# Class `hub.datasets.Flowers`

```python
hub.datasets.Flowers(
    transforms: Callable,
    mode: str = 'train')
```

Dataset for image classification. It contains 5 categories(roses, tulips, daisy, sunflowers, dandelion) and a total of 3667 pictures, of which 2914 are used for training, 382 are used for verification, and 371 are used for testing.

**Args**
* transforms(callmethod) : The method of preprocess images.
* mode(str): The mode for preparing dataset.

# Class `hub.datasets.MiniCOCO`

```python
hub.datasets.MiniCOCO(
    transforms: Callable,
    mode: str = 'train')
```

Dataset for Style transfer. The dataset contains 2001 images for training set and 200 images for testing set.They are derived form COCO2014. Meanwhile, it contains 21 different style pictures in file "21styles".

**Args**
* transforms(callmethod) : The method of preprocess images.
* mode(str): The mode for preparing dataset.

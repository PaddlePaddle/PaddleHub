# Class `hub.vision.Compose`

```python
hub.vision.Compose(
    transforms: Callable,
    to_rgb: bool = False)
```

Compose preprocessing operators for obtaining prepocessed data. The shape of input image for all operations is [H, W, C], where H is the image height, W is the image width, and C is the number of image channels.

**Args**
* transforms(callmethod) : The method of preprocess images.
* to_rgb(bool): Whether to transform the input from BGR mode to RGB mode, default is False.


# Class `hub.vision.RandomHorizontalFlip`

```python
hub.vision.RandomHorizontalFlip(prob: float = 0.5)
```

Randomly flip the image horizontally according to given probability.

**Args**

* prob(float): The probability for flipping the image horizontally, default is 0.5.


# Class `hub.vision.RandomVerticalFlip`

```python
hub.vision.RandomVerticalFlip(
    prob: float = 0.5)
```

Randomly flip the image vertically according to given probability.

**Args**

* prob(float): The probability for flipping the image vertically, default is 0.5.


# Class `hub.vision.Resize`

```python
hub.vision.Resize(
    target_size: Union[List[int], int], 
    interpolation: str = 'LINEAR')
```

Resize input image to target size.

**Args**

* target_size(List[int]|int]): Target image size.
* interpolation(str): Interpolation mode, default is 'LINEAR'. It support 6 modes: 'NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4' and 'RANDOM'.


# Class `hub.vision.ResizeByLong`

```python
hub.vision.ResizeByLong(long_size: int)
```

Resize the long side of the input image to the target size.

**Args**

* long_size(int|list[int]): The target size of long side.


# Class `hub.vision.ResizeRangeScaling`

```python
hub.vision.ResizeRangeScaling(
    min_value: int = 400, 
    max_value: int = 600)
```

Randomly select a targeted size to resize the image according to given range.

**Args**

* min_value(int): The minimum value for targeted size.
* max_value(int): The maximum value for targeted size.


# Class `hub.vision.ResizeStepScaling`

```python
hub.vision.ResizeStepScaling(
    min_scale_factor: float = 0.75, 
    max_scale_factor: float = 1.25,
    scale_step_size: float = 0.25)
```

Randomly select a scale factor to resize the image according to given range.

**Args**

* min_scale_factor(float): The minimum scale factor for targeted scale.
* max_scale_factor(float): The maximum scale factor for targeted scale.
* scale_step_size(float): Scale interval.


# Class `hub.vision.Normalize`

```python
hub.vision.Normalize(
    mean: list = [0.5, 0.5, 0.5], 
    std: list =[0.5, 0.5, 0.5])
```

Normalize the input image.

**Args**

* mean(list): Mean value for normalization.
* std(list): Standard deviation for normalization.


# Class `hub.vision.Padding`
 
 ```python
 hub.vision.Padding(
    target_size: Union[List[int], Tuple[int], int], 
    im_padding_value: list = [127.5, 127.5, 127.5])
 ```

 Padding input into targeted size according to specific padding value.

 **Args**

* target_size(Union[List[int], Tuple[int], int]): Targeted image size.
* im_padding_value(list): Border value for 3 channels, default is [127.5, 127.5, 127.5].


# Class `hub.vision.RandomPaddingCrop`
 
 ```python
 hub.vision.RandomPaddingCrop(
    crop_size(Union[List[int], Tuple[int], int]), 
    im_padding_value: list = [127.5, 127.5, 127.5])
 ```
 
 Padding input image if crop size is greater than image size. Otherwise, crop the input image to given size.

 **Args**

* crop_size(Union[List[int], Tuple[int], int]): Targeted image size.
* im_padding_value(list): Border value for 3 channels, default is [127.5, 127.5, 127.5].


# Class `hub.vision.RandomBlur`
 
 ```python
 hub.vision.RandomBlur(prob: float = 0.1)
 ```
 
 Random blur input image by Gaussian filter according to given probability.

 **Args**

* prob(float): The probability to blur the image, default is 0.1.


# Class `hub.vision.RandomRotation`
 
 ```python
 hub.vision.RandomRotation(
     max_rotation: float = 15., 
     im_padding_value: list = [127.5, 127.5, 127.5])
 ```
 
 Rotate the input image at random angle. The angle will not exceed to max_rotation.

 **Args**

* max_rotation(float): Upper bound of rotation angle.
* im_padding_value(list): Border value for 3 channels, default is [127.5, 127.5, 127.5].


# Class `hub.vision.RandomDistort`
 
 ```python
 hub.vision.RandomDistort(
     brightness_range: float = 0.5,
     brightness_prob: float = 0.5,
     contrast_range: float = 0.5,
     contrast_prob: float = 0.5,
     saturation_range: float = 0.5,
     saturation_prob: float = 0.5,
     hue_range: float= 18.,
     hue_prob: float= 0.5)
 ``` 
 
 Random adjust brightness, contrast, saturation and hue according to the given random range and probability, respectively.

 **Args**

* brightness_range(float): Boundary of brightness.
* brightness_prob(float): Probability for disturb the brightness of image.
* contrast_range(float): Boundary of contrast.
* contrast_prob(float): Probability for disturb the contrast of image.
* saturation_range(float): Boundary of saturation.
* saturation_prob(float): Probability for disturb the saturation of image.
* hue_range(float): Boundary of hue.
* hue_prob(float): Probability for disturb the hue of image.


# Class `hub.vision.RGB2LAB`
 
 ```python
 hub.vision.RGB2LAB()
 ```
 
 Convert color space from RGB to LAB.


# Class `hub.vision.LAB2RGB`
 
 ```python
 hub.vision.LAB2RGB()
 ```
 
 Convert color space from LAB to RGB.


# Class `hub.vision.CenterCrop`
 
 ```python
 hub.vision.CenterCrop(crop_size: int)
 ```
 
 Crop the middle part of the image to the specified size.

 **Args**

* crop_size(int): Target size for croped image.
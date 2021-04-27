==============
MiniCOCO
==============

.. code-block:: python

    class paddlehub.datasets.MiniCOCO(transform: Callable, mode: str = 'train'):

-----------------

   Dataset for Style transfer. The dataset contains 2001 images for training set and 200 images for testing set. They are derived form COCO2014. Meanwhile, it contains 21 different style pictures in file "21styles".

-----------------

* Args:
    * transform(Callable)
        The method of preprocess images.
    
    * mode(str)
        The mode for preparing dataset(train or test). Default to 'train'.
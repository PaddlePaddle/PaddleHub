==============
Canvas
==============

.. code-block:: python

    class paddlehub.datasets.Canvas(transform: Callable, mode: str = 'train'):

-----------------

   Dataset for colorization. It contains 1193 and 400 pictures for Monet and Vango paintings style, respectively. We collected data from https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/.

-----------------

* Args:
    * transform(Callable)
        The method of preprocess images.
    
    * mode(str)
        The mode for preparing dataset(train or test). Default to 'train'.
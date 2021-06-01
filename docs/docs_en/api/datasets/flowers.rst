==============
Flowers
==============

.. code-block:: python

    class paddlehub.datasets.Flowers(transform: Callable, mode: str = 'train'):

-----------------

   Flower classification dataset.

-----------------

* Args:
    * transform(Callable)
        The method of preprocess images.
    
    * mode(str)
        The mode for preparing dataset(train, test or val). Default to 'train'.
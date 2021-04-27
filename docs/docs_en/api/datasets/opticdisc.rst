==============
OpticDiscSeg
==============

.. code-block:: python

    class paddlehub.datasets.OpticDiscSeg(transform: Callable, mode: str = 'train'):

-----------------

   OpticDiscSeg dataset is extraced from iChallenge-AMD(https://ai.baidu.com/broad/subordinate?dataset=amd).
   
-----------------

* Args:
    * transform(Callable)
        The method of preprocess images.
    
    * mode(str)
        The mode for preparing dataset(train, test or val). Default to 'train'.
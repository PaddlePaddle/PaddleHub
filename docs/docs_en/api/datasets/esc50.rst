==============
ESC50
==============

.. code-block:: python

    class paddlehub.datasets.ESC50(mode: str = 'train', feat_type: str = 'mel'):)

-----------------

    The ESC-50 dataset is a labeled collection of 2000 environmental audio recordings suitable for benchmarking methods of environmental sound classification.
   
-----------------

* Args:
    * mode(:obj:`str`, `optional`, defaults to `train`):
        It identifies the dataset mode (train, test or dev).
    
    * feat_type(:obj:`str`, `optional`, defaults to `mel`):
        It identifies the input feature type (mel, or raw).
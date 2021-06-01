==============
MSRA_NER
==============

.. code-block:: python

    class paddlehub.datasets.MSRA_NER(tokenizer: Union[BertTokenizer, CustomTokenizer], max_seq_len: int = 128, mode: str = 'train'):

-----------------

    A set of manually annotated Chinese word-segmentation data and specifications for training and testing a Chinese word-segmentation system for research purposes.  For more information please refer to https://www.microsoft.com/en-us/download/details.aspx?id=52531

-----------------

* Args:
    * tokenizer(:obj:`BertTokenizer` or `CustomTokenizer`)
        It tokenizes the text and encodes the data as model needed.

    * max_seq_len(:obj:`int`, `optional`, defaults to :128)
        The maximum length (in number of tokens) for the inputs to the selected module, such as ernie, bert and so on.

    * mode(:obj:`str`, `optional`, defaults to `train`):
        It identifies the dataset mode (train, test or dev).
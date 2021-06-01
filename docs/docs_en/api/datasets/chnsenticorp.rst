==============
ChnSentiCorp
==============

.. code-block:: python

    class paddlehub.datasets.ChnSentiCorp(tokenizer: Union[BertTokenizer, CustomTokenizer], max_seq_len: int = 128, mode: str = 'train'):

-----------------

    ChnSentiCorp is a dataset for chinese sentiment classification, which was published by Tan Songbo at ICT of Chinese Academy of Sciences.

-----------------

* Args:
    * tokenizer(:obj:`BertTokenizer` or `CustomTokenizer`)
        It tokenizes the text and encodes the data as model needed.

    * max_seq_len(:obj:`int`, `optional`, defaults to :128)
        The maximum length (in number of tokens) for the inputs to the selected module, such as ernie, bert and so on.

    * mode(:obj:`str`, `optional`, defaults to `train`):
        It identifies the dataset mode (train, test or dev).
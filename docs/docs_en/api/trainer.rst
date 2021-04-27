==============
Trainer
==============

.. code-block:: python

    class paddlehub.Trainer(
        model: paddle.nn.Layer,
        optimizer: paddle.optimizer.Optimizer,
        use_gpu: bool = False,
        use_vdl: bool = True,
        checkpoint_dir: str = None,
        compare_metrics: Callable = None):

-----------------

   Model trainer.

-----------------

* Args:
    * model(paddle.nn.Layer)
        Model to train or evaluate.

    * optimizer(paddle.optimizer.Optimizer)
        Optimizer for loss.
        
    * use_gpu(bool)
        Whether to use gpu to run.

    * use_vdl(bool)
        Whether to use visualdl to record training data.

    * checkpoint_dir(str)
        Directory where the checkpoint is saved, and the trainer will restore the state and model parameters from the checkpoint.

    * compare_metrics(Callable)
        The method of comparing the model metrics. If not specified, the main metric return by `validation_step` will be used for comparison by default, the larger the value, the better the effect. This method will affect the saving of the best model. If the default behavior does not meet your requirements, please pass in a custom method.

**member functions**
=====================

train
------------------

    .. code-block:: python

        def train(
            train_dataset: paddle.io.Dataset,
            epochs: int = 1,
            batch_size: int = 1,
            num_workers: int = 0,
            eval_dataset: paddle.io.Dataset = None,
            log_interval: int = 10,
            save_interval: int = 10,
            collate_fn: Callable = None):

    Train a model with specific config.

    * Args:
        * train_dataset(paddle.io.Dataset)
            Dataset to train the model

        * epochs(int)
            Number of training loops, default is 1.

        * batch_size(int)
            Batch size of per step, default is 1.

        * num_workers(int)
            Number of subprocess to load data, default is 0.

        * eval_dataset(paddle.io.Dataset)
            The validation dataset, deafult is None. If set, the Trainer will execute evaluate function every `save_interval` epochs.
        
        * log_interval(int)
            Log the train infomation every `log_interval` steps.

        * save_interval(int)
            Save the checkpoint every `save_interval` epochs.

        * collate_fn(Callable)
            function to generate mini-batch data by merging the sample list. None for only stack each fields of sample in axis 0(same as :attr::`np.stack(..., axis=0)`). Default None.

evaluate
----------------------

    .. code-block:: python

        def evaluate(
            eval_dataset: paddle.io.Dataset,
            batch_size: int = 1,
            num_workers: int = 0,
            collate_fn: Callable = None):

    Run evaluation and returns metrics.

    * Args:
        * eval_dataset(paddle.io.Dataset)
            The validation dataset
        
        * batch_size(int)
            Batch size of per step, default is 1.

        * num_workers(int)
            Number of subprocess to load data, default is 0.
        
        * collate_fn(Callable)
            function to generate mini-batch data by merging the sample list. None for only stack each fields of sample in axis 0(same as :attr::`np.stack(..., axis=0)`). Default None.
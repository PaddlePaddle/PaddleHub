教程
==================

以下是关于PaddleHub的使用教程，介绍了命令行使用、如何自定义数据完成Finetune、如何自定义迁移任务、如何服务化部署预训练模型、如何获取ERNIE/BERT Embedding、如何用word2vec完成语义相似度计算、ULMFit优化策略介绍、如何使用超参优化AutoDL Finetuner、如何用Hook机制改写Task内置方法。


详细信息，参考以下教程：

..  toctree::
    :maxdepth: 1

    命令行工具<cmdintro>
    自定义数据<how_to_load_data>
    自定义任务<how_to_define_task>
    服务化部署<serving>
    文本Embedding服务<bert_service>
    语义相似度计算<sentence_sim>
    ULMFit优化策略<strategy_exp>
    超参优化<autofinetune>
    Hook机制<hook>
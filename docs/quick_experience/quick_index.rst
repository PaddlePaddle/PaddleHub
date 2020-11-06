快速体验
==================

PaddleHub有两种使用方式：Python代码调用和命令行调用。

命令行方式只需要一行命令即可快速体验PaddleHub提供的预训练模型的效果，是快速体验的绝佳选择；Python代码调用方式也仅需要三行代码，如果需要使用自己的数据Fine-tune并生成模型，则采用该方式。

命令行示例：
::
	$ hub run chinese_ocr_db_crnn_mobile --input_path test_ocr.jpg

Python代码示例：  
::
    import paddlehub as hub
    ocr = hub.Module(name="chinese_ocr_db_crnn_mobile")
    result = ocr.recognize_text(paths = "./test_image.jpg"], visualization=True, output_dir='ocr_output')

样例结果示例：  
  
.. image:: ../imgs/ocr_res.jpg


本章节提供这两种方法的快速体验方法，方便快速上手，同时提供了丰富的体验Demo，覆盖多场景领域，欢迎体验。具体使用时，当然还需要进一步了解详细的API接口参数、命令行参数解释，可参考后面的Python API接口和命令行参考章节。

..  toctree::
    :maxdepth: 1

    通过命令行调用方式使用PaddleHub<cmd_quick_run>
    通过Python代码调用方式使用PaddleHub<python_use_hub>
    PaddleHub更多体验Demos<more_demos>




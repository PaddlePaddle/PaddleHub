Quick Experience
==================

The PaddleHub is used in two modes: Python code execution and command line execution.

The command line mode requires only one line of command. It is perfect for quickly experiencing the effects of the pre-training models provided by PaddleHub. The Python code mode requires only three lines of codes. If you need to use your own data Fine-tune to generate models, you can use this method.

Example of command line:
::
	$ hub run chinese_ocr_db_crnn_mobile --input_path test_ocr.jpg

Example of Python Codes:
::
    import paddlehub as hub
    ocr = hub.Module(name="chinese_ocr_db_crnn_mobile")
    result = ocr.recognize_text(paths = "./test_image.jpg"], visualization=True, output_dir='ocr_output')

Example of results:

... image:: ... /imgs/ocr_res.jpg


This section provides a quick way to experience these two methods, so that you can get started quickly. It also provides a rich experience Demo, covering multiple scenarios. For more detailed explanations of the API interface parameters and command line parameters, refer to Python API Interface and Command Line.

...  toctree::
    :maxdepth: 1

    Command line execution: PaddleHub<cmd_quick_run>
    Python code execution: PaddleHub<python_use_hub>
    More Experience Demos of PaddleHub: <more_demos>

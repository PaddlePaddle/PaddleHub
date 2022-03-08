# Pneumonia_CT_LKM_PP_lung

|Module Name|Pneumonia_CT_LKM_PP_lung|
| :--- | :---: | 
|Category|Image segmentation|
|Network |-|
|Dataset|-|
|Fine-tuning supported or not|No|
|Module Size|35M|
|Data indicators|-|
|Latest update date|2021-02-26|


## I. Basic Information 


- ### Module Introduction

    - Pneumonia CT analysis model (Pneumonia-CT-LKM-PP) can efficiently complete the detection of lesions and outline the patient's CT images. Through post-processing codes, the number, volume, and lesions of lung lesions can be analyzed. This model has been fully trained by high-resolution and low-resolution CT image data, which can adapt to the examination data collected by different levels of CT imaging equipment. (This module is a submodule of Pneumonia_CT_LKM_PP.)

## II. Installation

- ### 1、Environmental Dependence

    - paddlepaddle >= 2.0.0

    - paddlehub >= 2.0.0

- ### 2、Installation

    - ```shell
      $ hub install Pneumonia_CT_LKM_PP_lung==1.0.0
      ```
      
    - In case of any problems during installation, please refer to:[Windows_Quickstart](../../../../docs/docs_ch/get_start/windows_quickstart.md)
    | [Linux_Quickstart](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_ch/get_start/mac_quickstart.md)  

## III. Module API Prediction

- ### 1、Prediction Code Example

    ```python
    import paddlehub as hub

    pneumonia = hub.Module(name="Pneumonia_CT_LKM_PP_lung")

    input_only_lesion_np_path = "/PATH/TO/ONLY_LESION_NP"
    input_both_lesion_np_path = "/PATH/TO/LESION_NP"
    input_both_lung_np_path = "/PATH/TO/LUNG_NP"

    # set input dict
    input_dict = {"image_np_path": [
                                    [input_only_lesion_np_path],
                                    [input_both_lesion_np_path, input_both_lung_np_path],
                                    ]}

    # execute predict and print the result
    results = pneumonia.segmentation(data=input_dict)
    for result in results:
        print(result)

    ```
   

- ### 2、API

  - ```python
    def segmentation(data)
    ```

    - Prediction API, used for CT analysis of pneumonia.

    - **Parameter**

        * data (dict): Key is "image_np_path", value is the list of results which contains lesion and lung segmentation masks. 
        

    - **Return**

        * result  (list\[dict\]): The list of recognition results, where each element is dict and each field is: 
            * input_lesion_np_path: Input path of lesion.
            * output_lesion_np: Segmentation result path of lesion.
            * input_lung_np_path: Input path of lung.
            * output_lung_np: Segmentation result path of lung.


## IV. Release Note

* 1.0.0

    First release

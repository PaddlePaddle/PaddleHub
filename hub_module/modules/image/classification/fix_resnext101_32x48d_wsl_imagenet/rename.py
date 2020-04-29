# coding=utf-8
import os

model_path = 'Fix_ResNeXt101_32x48d_wsl_pretrained'
model_path = os.path.join(os.getcwd(), model_path)

append_name = '@HUB_fix_resnext101_32x48d_wsl_imagenet@'
for file_name in os.listdir(model_path):
    new_file_name = append_name + file_name
    file_path = os.path.join(model_path, file_name)
    new_file_path = os.path.join(model_path, new_file_name)
    os.rename(file_path, new_file_path)

# coding=utf-8
import os

model_path = 'Res2Net101_vd_26w_4s_pretrained'
model_path = os.path.join(os.getcwd(), model_path)

append_name = '@HUB_res2net101_vd_26w_4s_imagenet@'
for file_name in os.listdir(model_path):
    new_file_name = append_name + file_name
    file_path = os.path.join(model_path, file_name)
    new_file_path = os.path.join(model_path, new_file_name)
    os.rename(file_path, new_file_path)

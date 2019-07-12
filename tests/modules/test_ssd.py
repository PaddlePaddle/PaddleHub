import paddlehub as hub
import os

ssd = hub.Module(name="ssd_mobilenet_v1_pascal")

base_dir = os.path.dirname(__file__)
test_img_path = os.path.join(base_dir, "resources", "test_img_cat.jpg")

# set input dict
input_dict = {"image": [test_img_path]}

# execute predict and print the result
results = ssd.object_detection(data=input_dict)
for result in results:
    print(result['path'])
    print(result['data'])

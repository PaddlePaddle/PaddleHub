import json

depth = [3, 4, 23, 3]
num_filters = [64, 128, 256, 512]

layer_index = 1
caffe_param_list = []

name_list = ['conv1']
params_list = []
name = name_list[0]
conv_w = name + '_weights'
caffe_conv_w = 'ConvNdBackward' + str(layer_index) + '_weights'
params_list.append(conv_w)
caffe_param_list.append(caffe_conv_w)

layer_index += 1

bn_name = "bn_" + name
caffe_bn_name = 'BatchNormBackward' + str(layer_index) + '_bn'
params_list.append(bn_name + '_scale')
params_list.append(bn_name + '_offset')
params_list.append(bn_name + '_mean')
params_list.append(bn_name + '_variance')

caffe_param_list.append(caffe_bn_name + '_scale')
caffe_param_list.append(caffe_bn_name + '_offset')
caffe_param_list.append(caffe_bn_name + '_mean')
caffe_param_list.append(caffe_bn_name + '_variance')

filter_input = 64

layer_index += 3

for block in range(len(depth)):
    for i in range(depth[block]):
        if block == 2:
            if i == 0:
                name = "res" + str(block + 2) + "a"
            else:
                name = "res" + str(block + 2) + "b" + str(i)
        else:
            name = "res" + str(block + 2) + chr(97 + i)

        name_list.append(name)

        for item in ['a', 'b', 'c']:
            name_branch = name + '_branch2' + item
            bn_name = 'bn' + name_branch[3:]
            params_list.append(name_branch + '_weights')
            params_list.append(bn_name + '_scale')
            params_list.append(bn_name + '_offset')
            params_list.append(bn_name + '_mean')
            params_list.append(bn_name + '_variance')

            caffe_name_branch = 'ConvNdBackward' + str(layer_index)
            caffe_param_list.append(caffe_name_branch + '_weights')

            layer_index += 1
            caffe_bn_name = 'BatchNormBackward' + str(layer_index) + '_bn'
            caffe_param_list.append(caffe_bn_name + '_scale')
            caffe_param_list.append(caffe_bn_name + '_offset')
            caffe_param_list.append(caffe_bn_name + '_mean')
            caffe_param_list.append(caffe_bn_name + '_variance')

            layer_index += 2

        stride = 2 if i == 0 and block != 0 else 1
        filter_num = num_filters[block]
        filter_output = filter_num * 4

        if (filter_output != filter_input) or (stride != 1):
            name_branch = name + '_branch1'

            print('filter_input {}, filter_output {}, stride {}, branch name {}'.format(
                filter_input, filter_output, stride, name_branch))
            bn_name = 'bn' + name_branch[3:]
            params_list.append(name_branch + '_weights')
            params_list.append(bn_name + '_scale')
            params_list.append(bn_name + '_offset')
            params_list.append(bn_name + '_mean')
            params_list.append(bn_name + '_variance')

            caffe_name_branch = 'ConvNdBackward' + str(layer_index)
            caffe_param_list.append(caffe_name_branch + '_weights')

            layer_index += 1
            caffe_bn_name = 'BatchNormBackward' + str(layer_index) + '_bn'
            caffe_param_list.append(caffe_bn_name + '_scale')
            caffe_param_list.append(caffe_bn_name + '_offset')
            caffe_param_list.append(caffe_bn_name + '_mean')
            caffe_param_list.append(caffe_bn_name + '_variance')

            layer_index += 3
        else:
            layer_index += 2

        filter_input = filter_output

map_dict = {}

for i in range(len(params_list)):
    print(params_list[i], caffe_param_list[i])
    map_dict[params_list[i]] = caffe_param_list[i]

json.dump(map_dict, open('name_map.json', 'w'))

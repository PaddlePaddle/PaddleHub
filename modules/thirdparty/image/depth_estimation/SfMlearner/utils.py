import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

def tensor2array(tensor, max_value=None, colormap='rainbow'):
    tensor = tensor.detach().cpu()
    if max_value is None:  # 计算最大值，后面用于归一化
        max_value = tensor.max().item()

    tensor = tensor.squeeze()
    if tensor.ndimension() == 2:  # 1张黑白图
        norm_array = tensor.numpy()/max_value  # 归一化
        # norm_array[norm_array == np.inf] = np.nan  # 去除无限大项
        array = COLORMAPS[colormap](norm_array).astype(np.float32)  # [height, width, 4]
        array = array.transpose(2, 0, 1)[:3]  # 取RGB这3个维度

    elif tensor.ndimension() == 3:  # 1张RGB图
        assert(tensor.shape[0] == 3)
        array = 0.5 + tensor.numpy()*0.5  # 取值范围从[-1, 1]调整到[0, 1]
    return array


def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
    # Construct the list colormap, with interpolated values for higer resolution
    # For a linear segmented colormap, you can just specify the number of point in
    # cm.get_cmap(name, lutsize) with the parameter lutsize
    x = np.linspace(0, 1, low_res_cmap.N)
    low_res = low_res_cmap(x)
    new_x = np.linspace(0, max_value, resolution)
    high_res = np.stack([np.interp(new_x, x, low_res[:, i]) for i in range(low_res.shape[1])], axis=1)
    return ListedColormap(high_res)


def opencv_rainbow(resolution=1000):
    # Construct the opencv equivalent of Rainbow
    opencv_rainbow_data = (
        (0.000, (1.00, 0.00, 0.00)),
        (0.400, (1.00, 1.00, 0.00)),
        (0.600, (0.00, 1.00, 0.00)),
        (0.800, (0.00, 0.00, 1.00)),
        (1.000, (0.60, 0.00, 1.00))
    )
    return LinearSegmentedColormap.from_list('opencv_rainbow', opencv_rainbow_data, resolution)


COLORMAPS = {'rainbow': opencv_rainbow(),
             'magma': high_res_colormap(cm.get_cmap('magma')),
             'bone': cm.get_cmap('bone', 10000)}
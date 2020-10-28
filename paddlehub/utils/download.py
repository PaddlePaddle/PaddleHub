import os

from paddlehub.env import DATA_HOME
from paddle.utils.download import get_path_from_url


def download_data(url):
    save_name = os.path.basename(url).split('.')[0]
    output_path = os.path.join(DATA_HOME, save_name)

    if not os.path.exists(output_path):
        get_path_from_url(url, DATA_HOME)

    def _wrapper(Dataset):
        return Dataset

    return _wrapper

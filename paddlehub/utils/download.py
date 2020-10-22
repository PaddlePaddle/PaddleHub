import os

from paddlehub.utils.xarfile import XarFile
from paddlehub.utils import log, utils
from paddlehub.env import DATA_HOME


def download_data(name, url):
    data_path = os.path.join(DATA_HOME, name)
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    if not os.listdir(data_path):
        with log.ProgressBar('Download {}'.format(url)) as bar:
            for file, ds, ts in utils.download_with_progress(url, DATA_HOME):
                bar.update(float(ds) / ts)
        tar_file = data_path + '.tar.gz'
        tfp = XarFile(tar_file, 'r', 'tar.gz')
        tfp.extractall(DATA_HOME)
        print('Extract dataset...')
        os.remove(tar_file)

    def _wrapper(Dataset):
        return Dataset

    return _wrapper

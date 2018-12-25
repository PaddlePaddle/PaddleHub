from urllib.request import urlretrieve
from tqdm import tqdm

import os
import tempfile
"""
tqdm prograss hook
"""


class TqdmProgress(tqdm):
    last_block = 0

    def update_to(self, block_num=1, block_size=1, total_size=None):
        '''
        block_num  : int, optional
            到目前为止传输的块 [default: 1].
        block_size : int, optional
            每个块的大小 (in tqdm units) [default: 1].
        total_size : int, optional
            文件总大小 (in tqdm units). 如果[default: None]保持不变.
        '''
        if total_size is not None:
            self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


class DownloadManager(object):
    def __init__(self):
        self.dst_path = tempfile.mkstemp()

    def download(self, link, dst_path):
        file_name = link.split("/")[-1]
        if dst_path is not None:
            self.dst_path = dst_path
        if not os.path.exists(self.dst_path):
            os.makedirs(self.dst_path)
        file_path = os.path.join(self.dst_path, file_name)
        print("download filepath", file_path)

        with TqdmProgress(
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                miniters=1,
                desc=file_name) as progress:
            path, header = urlretrieve(
                link,
                filename=file_path,
                reporthook=progress.update_to,
                data=None)

            return path

    def _extract_file(self, tgz, tarinfo, dst_path, buffer_size=10 << 20):
        """Extracts 'tarinfo' from 'tgz' and writes to 'dst_path'."""
        src = tgz.extractfile(tarinfo)
        dst = tf.gfile.GFile(dst_path, "wb")
        while 1:
            buf = src.read(buffer_size)
            if not buf:
                break
            dst.write(buf)
            self._log_progress(len(buf))
        dst.close()
        src.close()

    def download_and_uncompress(self, link, dst_path):
        file_name = self.download(link, dst_path)
        print(file_name)


if __name__ == "__main__":
    link = "ftp://nj03-rp-m22nlp062.nj03.baidu.com//home/disk0/chenzeyu01/movie/movie_summary.txt"
    dl = DownloadManager()
    dl.download_and_uncompress(link, "./tmp")

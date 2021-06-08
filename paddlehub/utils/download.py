# coding:utf-8
# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import filelock

import paddlehub.env as hubenv
from paddle.utils.download import get_path_from_url
from paddlehub.utils import log, utils, xarfile


def download_data(url):
    def _wrapper(Dataset):
        def _check_download():
            save_name = os.path.basename(url).split('.')[0]
            output_path = os.path.join(hubenv.DATA_HOME, save_name)
            lock = filelock.FileLock(os.path.join(hubenv.TMP_HOME, save_name))
            with lock:
                if not os.path.exists(output_path):
                    default_downloader.download_file_and_uncompress(url, hubenv.DATA_HOME, True)

        class WrapperDataset(Dataset):
            def __new__(cls, *args, **kwargs):
                _check_download()
                return super(WrapperDataset, cls).__new__(cls)

        return WrapperDataset

    return _wrapper


class Downloader:
    def download_file_and_uncompress(self, url: str, save_path: str, print_progress: bool):
        with utils.generate_tempdir() as _dir:
            if print_progress:
                with log.ProgressBar('Download {}'.format(url)) as bar:
                    for path, ds, ts in utils.download_with_progress(url=url, path=_dir):
                        bar.update(float(ds) / ts)
            else:
                path = utils.download(url=url, path=_dir)

            if print_progress:
                with log.ProgressBar('Decompress {}'.format(path)) as bar:
                    for path, ds, ts in xarfile.unarchive_with_progress(name=path, path=save_path):
                        bar.update(float(ds) / ts)
            else:
                path = xarfile.unarchive(name=path, path=save_path)


default_downloader = Downloader()

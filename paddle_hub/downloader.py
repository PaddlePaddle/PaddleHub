#   Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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
# coding=utf-8

from __future__ import print_function
from __future__ import division
from __future__ import print_function

import os
import sys
import hashlib
import requests
import tempfile
import tarfile

__all__ = ['MODULE_HOME', 'download', 'md5file', 'download_and_uncompress']

# TODO(ZeyuChen) add environment varialble to set MODULE_HOME
MODULE_HOME = os.path.expanduser('~/.cache/paddle/module')


# When running unit tests, there could be multiple processes that
# trying to create MODULE_HOME directory simultaneously, so we cannot
# use a if condition to check for the existence of the directory;
# instead, we use the filesystem as the synchronization mechanism by
# catching returned errors.
def must_mkdirs(path):
    try:
        os.makedirs(MODULE_HOME)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass


def md5file(fname):
    hash_md5 = hashlib.md5()
    f = open(fname, "rb")
    for chunk in iter(lambda: f.read(4096), b""):
        hash_md5.update(chunk)
    f.close()
    return hash_md5.hexdigest()


def download_and_uncompress(url, save_name=None):
    module_name = url.split("/")[-2]
    dirname = os.path.join(MODULE_HOME, module_name)
    print("download to dir", dirname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    #TODO(ZeyuChen) add download md5 file to verify file completeness
    file_name = os.path.join(
        dirname,
        url.split('/')[-1] if save_name is None else save_name)

    retry = 0
    retry_limit = 3
    while not (os.path.exists(file_name)):
        if os.path.exists(file_name):
            print("file md5", md5file(file_name))
        if retry < retry_limit:
            retry += 1
        else:
            raise RuntimeError(
                "Cannot download {0} within retry limit {1}".format(
                    url, retry_limit))
        print("Cache file %s not found, downloading %s" % (file_name, url))
        r = requests.get(url, stream=True)
        total_length = r.headers.get('content-length')

        if total_length is None:
            with open(file_name, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        else:
            #TODO(ZeyuChen) upgrade to tqdm process
            with open(file_name, 'wb') as f:
                dl = 0
                total_length = int(total_length)
                for data in r.iter_content(chunk_size=4096):
                    dl += len(data)
                    f.write(data)
                    done = int(50 * dl / total_length)
                    sys.stdout.write(
                        "\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
                    sys.stdout.flush()

    print("file download completed!", file_name)
    #TODO(ZeyuChen) add md5 check error and file incompleted error, then raise
    # them and catch them
    with tarfile.open(file_name, "r:gz") as tar:
        file_names = tar.getnames()
        print(file_names)
        module_dir = os.path.join(dirname, file_names[0])
        for file_name in file_names:
            tar.extract(file_name, dirname)

    return module_name, module_dir


if __name__ == "__main__":
    # TODO(ZeyuChen) add unit test
    link = "http://paddlehub.bj.bcebos.com/word2vec/word2vec-dim16-simple-example-1.tar.gz"

    module_path = download_and_uncompress(link)
    print("module path", module_path)

    # dl = DownloadManager()
    # dl.download_and_uncompress(link, "./tmp")

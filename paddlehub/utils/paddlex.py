# coding:utf-8
# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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
import shutil

from paddlehub.server.server import module_server, CacheUpdater
from paddlehub.utils import log, utils, xarfile


class ResourceNotFoundError(Exception):
    def __init__(self, name: str, version: str = None):
        self.name = name
        self.version = version

    def __str__(self):
        if not self.version:
            tips = 'No resource named {} was found'.format(self.name)
        else:
            tips = 'No resource named {}-{} was found'.format(self.name, self.version)
        return tips


def download(name: str, save_path: str, version: str = None):
    '''The download interface provided to PaddleX for downloading the specified model and resource files.'''

    CacheUpdater("x_download", name, version).start()

    file = os.path.join(save_path, name)
    file = os.path.realpath(file)
    if os.path.exists(file):
        return

    resources = module_server.search_resource(name=name, version=version, type='Model')
    if not resources:
        raise ResourceNotFoundError(name, version)

    for item in resources:
        if item['name'] == name and utils.Version(item['version']).match(version):
            url = item['url']
            break
    else:
        raise ResourceNotFoundError(name, version)

    with utils.generate_tempdir() as _dir:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with log.ProgressBar('Download {}'.format(url)) as _bar:
            for savefile, dsize, tsize in utils.download_with_progress(url, _dir):
                _bar.update(float(dsize / tsize))

        if xarfile.is_xarfile(savefile):
            with log.ProgressBar('Decompress {}'.format(savefile)) as _bar:
                for savefile, usize, tsize in xarfile.unarchive_with_progress(savefile, _dir):
                    _bar.update(float(usize / tsize))

                savefile = os.path.join(_dir, savefile.split(os.sep)[0])

        shutil.move(savefile, file)

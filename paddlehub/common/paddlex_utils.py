#coding:utf-8
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
import tarfile
import shutil
from paddlehub.common import tmp_dir
from paddlehub.common.hub_server import CacheUpdater
from paddlehub.common.downloader import default_downloader
import paddlehub as hub


class ResourceNotFoundError(Exception):
    def __init__(self, name, version=None):
        self.name = name
        self.version = version

    def __str__(self):
        if self.version:
            tips = 'No resource named {} was found'.format(self.name)
        else:
            tips = 'No resource named {}-{} was found'.format(
                self.name, self.version)
        return tips


class ServerConnectionError(Exception):
    def __str__(self):
        tips = 'Can\'t connect to Hub Server:{}'.format(
            hub.HubServer().server_url[0])
        return tips


def download(name,
             save_path,
             version=None,
             decompress=True,
             resource_type='Model',
             extra={}):
    file = os.path.join(save_path, name)
    file = os.path.realpath(file)
    if os.path.exists(file):
        return

    if not hub.HubServer()._server_check():
        raise ServerConnectionError

    search_result = hub.HubServer().get_resource_url(
        name, resource_type=resource_type, version=version, extra=extra)

    if not search_result:
        raise ResourceNotFoundError(name, version)
    CacheUpdater("x_download", name, version).start()
    url = search_result['url']

    with tmp_dir() as _dir:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        _, _, savefile = default_downloader.download_file(
            url=url, save_path=_dir, print_progress=True)
        if tarfile.is_tarfile(savefile) and decompress:
            _, _, savefile = default_downloader.uncompress(
                file=savefile, print_progress=True)
        shutil.move(savefile, file)

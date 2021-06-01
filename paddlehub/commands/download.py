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

from typing import List

import paddlehub as hub
from paddlehub.commands import register
from paddlehub.server import module_server
from paddlehub.utils import utils, log
from paddlehub.server.server import CacheUpdater


@register(name='hub.download', description='Download PaddlePaddle pretrained module files.')
class DownloadCommand:
    def execute(self, argv: List) -> bool:
        if not argv:
            print("ERROR: You must give at least one module to download.")
            return False

        for _arg in argv:
            result = module_server.search_module(_arg)
            CacheUpdater("hub_download", _arg).start()
            if result:
                url = result[0]['url']
                with log.ProgressBar('Download {}'.format(url)) as bar:
                    for file, ds, ts in utils.download_with_progress(url):
                        bar.update(float(ds) / ts)
            else:
                print('ERROR: Could not find a HubModule named {}'.format(_arg))
        return True

#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

from tqdm import tqdm
from paddlehub.common.logger import logger
from paddlehub.common.dir import MODULE_HOME


def _fetch_from_remote(url, force_download=False):
    import tempfile, requests, tarfile
    cached_dir = os.path.join(MODULE_HOME, "ernie_for_gen")
    if force_download or not os.path.exists(cached_dir):
        with tempfile.NamedTemporaryFile() as f:
            #url = 'https://ernie.bj.bcebos.com/ERNIE_stable.tgz'
            r = requests.get(url, stream=True)
            total_len = int(r.headers.get('content-length'))
            for chunk in tqdm(
                    r.iter_content(chunk_size=1024),
                    total=total_len // 1024,
                    desc='downloading %s' % url,
                    unit='KB'):
                if chunk:
                    f.write(chunk)
                    f.flush()
            logger.debug('extacting... to %s' % f.name)
            with tarfile.open(f.name) as tf:
                tf.extractall(path=cached_dir)
    logger.debug('%s cached in %s' % (url, cached_dir))
    return cached_dir


def add_docstring(doc):
    def func(f):
        f.__doc__ += ('\n======other docs from supper class ======\n%s' % doc)
        return f

    return func

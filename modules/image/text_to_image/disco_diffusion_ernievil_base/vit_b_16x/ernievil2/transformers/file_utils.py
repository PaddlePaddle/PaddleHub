from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import time
from pathlib import Path

import six
from tqdm import tqdm
if six.PY2:
    from pathlib2 import Path
else:
    from pathlib import Path

log = logging.getLogger(__name__)


def _fetch_from_remote(url, force_download=False, cached_dir='~/.paddle-ernie-cache'):
    import hashlib, tempfile, requests, tarfile
    sig = hashlib.md5(url.encode('utf8')).hexdigest()
    cached_dir = Path(cached_dir).expanduser()
    try:
        cached_dir.mkdir()
    except OSError:
        pass
    cached_dir_model = cached_dir / sig
    from filelock import FileLock
    with FileLock(str(cached_dir_model) + '.lock'):
        donefile = cached_dir_model / 'done'
        if (not force_download) and donefile.exists():
            log.debug('%s cached in %s' % (url, cached_dir_model))
            return cached_dir_model
        cached_dir_model.mkdir(exist_ok=True)
        tmpfile = cached_dir_model / 'tmp'
        with tmpfile.open('wb') as f:
            r = requests.get(url, stream=True)
            total_len = int(r.headers.get('content-length'))
            for chunk in tqdm(r.iter_content(chunk_size=1024),
                              total=total_len // 1024,
                              desc='downloading %s' % url,
                              unit='KB'):
                if chunk:
                    f.write(chunk)
                    f.flush()
            log.debug('extacting... to %s' % tmpfile)
            with tarfile.open(tmpfile.as_posix()) as tf:
                tf.extractall(path=str(cached_dir_model))
            donefile.touch()
        os.remove(tmpfile.as_posix())

    return cached_dir_model


def add_docstring(doc):

    def func(f):
        f.__doc__ += ('\n======other docs from supper class ======\n%s' % doc)
        return f

    return func

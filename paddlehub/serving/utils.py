# coding: utf-8
import hashlib
import time


def gen_md5(src_str):
    md5_get = hashlib.md5()
    md5_src = str(time.time()) + str(src_str)
    md5_get.update(md5_src.encode("utf-8"))
    md5_value = md5_get.hexdigest()
    return md5_value

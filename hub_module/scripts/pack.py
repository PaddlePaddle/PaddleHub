# coding:utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import tempfile
import tarfile
import shutil
import yaml
import re

import paddlehub as hub

from downloader import downloader

PACK_PATH = os.path.dirname(os.path.realpath(__file__))
MODULE_BASE_PATH = os.path.join(PACK_PATH, "..")


def parse_args():
    parser = argparse.ArgumentParser(description='packing PaddleHub Module')
    parser.add_argument(
        '--config',
        dest='config',
        help='Config file for module config',
        default=None,
        type=str)
    return parser.parse_args()


def package_module(config):
    with tempfile.TemporaryDirectory(dir=".") as _dir:
        directory = os.path.join(MODULE_BASE_PATH, config["dir"])
        name = config['name'].replace('-', '_')
        dest = os.path.join(_dir, name)
        shutil.copytree(directory, dest)
        for resource in config.get("resources", {}):
            if resource.get("uncompress", False):
                _, _, file = downloader.download_file_and_uncompress(
                    url=resource["url"], save_path=dest, print_progress=True)
            else:
                _, _, file = downloader.download_file(
                    url=resource["url"], save_path=dest, print_progress=True)

            dest_path = os.path.join(dest, resource["dest"])
            if resource["dest"] != ".":
                if os.path.realpath(dest_path) != os.path.realpath(file):
                    shutil.move(file, dest_path)

        tar_filter = lambda tarinfo: None if any([
            exclude_file_name in tarinfo.name.replace(name + os.sep, "")
            for exclude_file_name in config.get("exclude", [])
        ]) else tarinfo

        with open(os.path.join(directory, "module.py")) as file:
            file_content = file.read()
            file_content = file_content.replace('\n',
                                                '').replace(' ', '').replace(
                                                    '"', '').replace("'", '')
            module_info = re.findall('@moduleinfo\(.*?\)',
                                     file_content)[0].replace(
                                         '@moduleinfo(', '').replace(')', '')
            module_info = module_info.split(',')
            for item in module_info:
                if item.startswith('version'):
                    module_version = item.split('=')[1].replace(',', '')
                if item.startswith('name'):
                    module_name = item.split('=')[1].replace(',', '')
        package = "{}_{}.tar.gz".format(module_name, module_version)
        with tarfile.open(package, "w:gz") as tar:
            tar.add(
                dest, arcname=os.path.basename(module_name), filter=tar_filter)


def main(args):
    with open(args.config, "r") as file:
        config = yaml.load(file.read(), Loader=yaml.FullLoader)

    package_module(config)


if __name__ == "__main__":
    main(parse_args())

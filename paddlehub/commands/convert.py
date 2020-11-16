#coding:utf-8
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
import argparse
import os
import time
import tarfile
import shutil
from string import Template

from paddlehub.env import TMP_HOME as tmp_dir
from paddlehub.commands import register
from paddlehub.utils.xarfile import XarFile
from paddlehub.server.server import CacheUpdater

INIT_FILE = '__init__.py'
MODULE_FILE = 'module.py'
SERVING_FILE = 'serving_client_demo.py'
TMPL_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tmpl')


@register(name='hub.convert', description='Convert model to PaddleHub-Module.')
class ConvertCommand:
    def __init__(self):
        super(ConvertCommand, self).__init__()
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('command')
        self.parser.add_argument('--module_name', '-n')
        self.parser.add_argument('--module_version', '-v', nargs='?', default='1.0.0')
        self.parser.add_argument('--model_dir', '-d')
        self.parser.add_argument('--output_dir', '-o')

    def create_module_tar(self):
        if not os.path.exists(self.dest):
            os.makedirs(self.dest)
        tar_file = os.path.join(self.dest, '{}.tar.gz'.format(self.module))
        tfp = XarFile(tar_file, 'w', 'tar.gz')
        tfp.add(self.dest, self.module, False)
        for root, dir, files in os.walk(self.src):
            for file in files:
                fullpath = os.path.join(root, file)
                arcname = os.path.join(self.module, 'assets', file)
                tfp.add(fullpath, arcname=arcname)

        tfp.add(name=self.model_file, arcname=os.path.join(self.module, MODULE_FILE))
        tfp.add(name=self.serving_file, arcname=os.path.join(self.module, SERVING_FILE))
        tfp.add(name=self.init_file, arcname=os.path.join(self.module, INIT_FILE))

    def create_module_py(self):
        template_file = open(os.path.join(TMPL_DIR, 'x_model.tmpl'), 'r', encoding='utf-8')
        tmpl = Template(template_file.read())
        lines = []

        lines.append(
            tmpl.substitute(
                NAME="'{}'".format(self.module),
                TYPE="'CV'",
                AUTHOR="'Baidu'",
                SUMMARY="''",
                VERSION="'{}'".format(self.version),
                EMAIL="''"))
        self.model_file = os.path.join(self._tmp_dir, MODULE_FILE)

        with open(self.model_file, 'w', encoding='utf-8') as fp:
            fp.writelines(lines)

    def create_init_py(self):
        self.init_file = os.path.join(self._tmp_dir, INIT_FILE)
        if os.path.exists(self.init_file):
            return
        shutil.copyfile(os.path.join(TMPL_DIR, 'init_py.tmpl'), self.init_file)

    def create_serving_demo_py(self):
        template_file = open(os.path.join(TMPL_DIR, 'serving_demo.tmpl'), 'r', encoding='utf-8')
        tmpl = Template(template_file.read())
        lines = []

        lines.append(tmpl.substitute(MODULE_NAME=self.module))
        self.serving_file = os.path.join(self._tmp_dir, SERVING_FILE)
        with open(self.serving_file, 'w', encoding='utf-8') as fp:
            fp.writelines(lines)

    @staticmethod
    def show_help():
        str = "convert --module <module> [--version <version>] --dest dest_dir --src srd_dir\n"
        str += "\tConvert model to PaddleHub-Module.\n"
        str += "--model_dir\n"
        str += "\tDir of model you want to export.\n"
        str += "--module_name:\n"
        str += "\tSet name of module.\n"
        str += "--module_version\n"
        str += "\tSet version of module, default is `1.0.0`.\n"
        str += "--output_dir\n"
        str += "\tDir to save PaddleHub-Module after exporting, default is `.`.\n"
        print(str)
        return

    def execute(self, argv):
        args = self.parser.parse_args()

        if not args.module_name or not args.model_dir:
            ConvertCommand.show_help()
            return False
        self.module = args.module_name
        self.version = args.module_version if args.module_version is not None else '1.0.0'
        CacheUpdater("hub_convert", module=self.module, version=self.version).start()
        self.src = args.model_dir
        if not os.path.isdir(self.src):
            print('`{}` is not exists or not a directory path'.format(self.src))
            return False
        self.dest = args.output_dir if args.output_dir is not None else os.path.join('{}_{}'.format(
            self.module, str(time.time())))
        os.makedirs(self.dest)

        self._tmp_dir = tmp_dir
        self.create_module_py()
        self.create_init_py()
        self.create_serving_demo_py()
        self.create_module_tar()

        print('The converted module is stored in `{}`.'.format(self.dest))

        return True

# -*- coding:utf-8 -**
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""ERNIE"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging

import six


class ErnieConfig(object):
    """parse ernie config"""

    def __init__(self, config_path):
        """
        :param config_path:
        """
        self._config_dict = self._parse(config_path)

    def _parse(self, config_path):
        """
        :param config_path:
        :return:
        """
        try:
            with open(config_path, 'r') as json_file:
                config_dict = json.load(json_file)
        except Exception:
            raise IOError("Error in parsing Ernie model config file '%s'" % config_path)
        else:
            return config_dict

    def __getitem__(self, key):
        """
        :param key:
        :return:
        """
        return self._config_dict.get(key, None)

    def has(self, key):
        """
        :param key:
        :return:
        """
        if key in self._config_dict:
            return True
        return False

    def get(self, key, default_value):
        """
        :param key,default_value:
        :retrun:
        """
        if key in self._config_dict:
            return self._config_dict[key]
        else:
            return default_value

    def print_config(self):
        """
        :return:
        """
        for arg, value in sorted(six.iteritems(self._config_dict)):
            logging.info('%s: %s' % (arg, value))
        logging.info('------------------------------------------------')

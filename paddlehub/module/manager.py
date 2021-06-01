# coding:utf-8
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

import os
import shutil

import sys
from collections import OrderedDict
from typing import List

import filelock

from paddlehub.env import MODULE_HOME, TMP_HOME
from paddlehub.module.module import Module as HubModule
from paddlehub.server import module_server
from paddlehub.utils import xarfile, log, utils, pypi


class HubModuleNotFoundError(Exception):
    def __init__(self, name: str, info: dict = None, version: str = None, source: str = None):
        self.name = name
        self.version = version
        self.info = info
        self.source = source

    def __str__(self):
        msg = '{}'.format(self.name)
        if self.version:
            msg += '-{}'.format(self.version)

        if self.source:
            msg += ' from {}'.format(self.source)

        tips = 'No HubModule named {} was found'.format(log.FormattedText(text=msg, color='red'))

        if self.info:
            sort_infos = sorted(self.info.items(), key=lambda x: utils.Version(x[0]))

            table = log.Table()
            table.append(
                *['Name', 'Version', 'PaddlePaddle Version Required', 'PaddleHub Version Required'],
                widths=[15, 10, 35, 35],
                aligns=['^', '^', '^', '^'],
                colors=['cyan', 'cyan', 'cyan', 'cyan'])

            for _ver, info in sort_infos:
                paddle_version = 'Any' if not info['paddle_version'] else ', '.join(info['paddle_version'])
                hub_version = 'Any' if not info['hub_version'] else ', '.join(info['hub_version'])
                table.append(self.name, _ver, paddle_version, hub_version, aligns=['^', '^', '^', '^'])

            tips += ':\n{}'.format(table)
        return tips


class EnvironmentMismatchError(Exception):
    def __init__(self, name: str, info: dict, version: str = None):
        self.name = name
        self.version = version
        self.info = info

    def __str__(self):
        msg = '{}'.format(self.name)
        if self.version:
            msg += '-{}'.format(self.version)

        tips = '{} cannot be installed because some conditions are not met'.format(
            log.FormattedText(text=msg, color='red'))

        if self.info:
            sort_infos = sorted(self.info.items(), key=lambda x: utils.Version(x[0]))

            table = log.Table()
            table.append(
                *['Name', 'Version', 'PaddlePaddle Version Required', 'PaddleHub Version Required'],
                widths=[15, 10, 35, 35],
                aligns=['^', '^', '^', '^'],
                colors=['cyan', 'cyan', 'cyan', 'cyan'])

            import paddle
            import paddlehub

            for _ver, info in sort_infos:
                paddle_version = 'Any' if not info['paddle_version'] else ', '.join(info['paddle_version'])
                for version in info['paddle_version']:
                    if not utils.Version(paddle.__version__).match(version):
                        paddle_version = '{}(Mismatch)'.format(paddle_version)
                        break

                hub_version = 'Any' if not info['hub_version'] else ', '.join(info['hub_version'])
                for version in info['hub_version']:
                    if not utils.Version(paddlehub.__version__).match(version):
                        hub_version = '{}(Mismatch)'.format(hub_version)
                        break

                table.append(self.name, _ver, paddle_version, hub_version, aligns=['^', '^', '^', '^'])

            tips += ':\n{}'.format(table)
        return tips


class LocalModuleManager(object):
    '''
    LocalModuleManager is used to manage PaddleHub's local Module, which supports the installation, uninstallation,
    and search of HubModule. LocalModuleManager is a singleton object related to the path, in other words, when the
    LocalModuleManager object of the same home directory is generated multiple times, the same object is returned.

    Args:
        home (str): The directory where PaddleHub modules are stored, the default is ~/.paddlehub/modules
    '''
    _instance_map = {}

    def __new__(cls, home: str = MODULE_HOME):
        home = MODULE_HOME if not home else home
        if home in cls._instance_map:
            return cls._instance_map[home]
        cls._instance_map[home] = super(LocalModuleManager, cls).__new__(cls)
        return cls._instance_map[home]

    def __init__(self, home: str = None):
        home = MODULE_HOME if not home else home
        self.home = home
        self._local_modules = OrderedDict()

        # Most HubModule can be regarded as a python package, so we need to add the home
        # directory to sys.path
        if not home in sys.path:
            sys.path.insert(0, home)

    def _get_normalized_path(self, name: str) -> str:
        return os.path.join(self.home, self._get_normalized_name(name))

    def _get_normalized_name(self, name: str) -> str:
        # Some HubModules contain '-'  in name (eg roberta_wwm_ext_chinese_L-3_H-1024_A-16).
        # Replace '-' with '_' to comply with python naming conventions.
        return name.replace('-', '_')

    def install(self,
                *,
                name: str = None,
                directory: str = None,
                archive: str = None,
                url: str = None,
                version: str = None,
                source: str = None,
                update: bool = False,
                branch: str = None,
                ignore_env_mismatch: bool = False) -> HubModule:
        '''
        Install a HubModule from name or directory or archive file or url. When installing with the name parameter, if a
        module that meets the conditions (both name and version) already installed, the installation step will be skipped.
        When installing with other parameter, The locally installed modules will be uninstalled.

        Args:
            name                (str|optional): module name to install
            directory           (str|optional): directory containing  module code
            archive             (str|optional): archive file containing  module code
            url                 (str|optional): url points to a archive file containing module code
            version             (str|optional): module version, use with name parameter
            source              (str|optional): source containing module code, use with name paramete
            ignore_env_mismatch (str|optional): Whether to ignore the environment mismatch when installing the Module.
        '''
        if name:

            lock = filelock.FileLock(os.path.join(TMP_HOME, name))
            with lock:
                hub_module_cls = self.search(name, source, branch)
                if hub_module_cls and hub_module_cls.version.match(version):
                    directory = self._get_normalized_path(hub_module_cls.name)
                    if version:
                        msg = 'Module {}-{} already installed in {}'.format(hub_module_cls.name, hub_module_cls.version,
                                                                            directory)
                    else:
                        msg = 'Module {} already installed in {}'.format(hub_module_cls.name, directory)
                    log.logger.info(msg)
                    return hub_module_cls
                if source:
                    return self._install_from_source(name, version, source, update, branch)
                return self._install_from_name(name, version, ignore_env_mismatch)
        elif directory:
            return self._install_from_directory(directory)
        elif archive:
            return self._install_from_archive(archive)
        elif url:
            return self._install_from_url(url)
        else:
            raise RuntimeError('Attempt to install a module, but no parameters were specified.')

    def uninstall(self, name: str) -> bool:
        '''Return True if uninstall successfully else False'''
        if not os.path.exists(self._get_normalized_path(name)):
            log.logger.info('{} is not installed'.format(name))
            return False

        shutil.rmtree(self._get_normalized_path(name))
        if name in self._local_modules:
            log.logger.info('Successfully uninstalled {}-{}'.format(name, self._local_modules[name].version))
            self._local_modules.pop(name)
        else:
            log.logger.info('Successfully uninstalled {}'.format(name))
        return True

    def search(self, name: str, source: str = None, branch: str = None) -> HubModule:
        '''Return HubModule If a HubModule with a specific name is found, otherwise None.'''
        module = None

        if name in self._local_modules:
            module = self._local_modules[name]
        else:
            module_dir = self._get_normalized_path(name)
            if os.path.exists(module_dir):
                try:
                    module = self._local_modules[name] = HubModule.load(module_dir)
                except:
                    utils.record_exception('An error was encountered while loading {}'.format(name))

        if not module:
            return None

        if source and source != module.source:
            return None

        if branch and branch != module.branch:
            return None

        return module

    def list(self) -> List[HubModule]:
        '''List all installed HubModule.'''
        for subdir in os.listdir(self.home):
            fulldir = os.path.join(self.home, subdir)

            try:
                self._local_modules[subdir] = HubModule.load(fulldir)
            except:
                utils.record_exception('An error was encountered while loading {}'.format(subdir))

        return [module for module in self._local_modules.values()]

    def _install_from_url(self, url: str) -> HubModule:
        '''Install HubModule from url'''
        with utils.generate_tempdir() as _tdir:
            with log.ProgressBar('Download {}'.format(url)) as bar:
                for file, ds, ts in utils.download_with_progress(url, _tdir):
                    bar.update(float(ds) / ts)

            return self._install_from_archive(file)

    def _install_from_name(self, name: str, version: str = None, ignore_env_mismatch: bool = False) -> HubModule:
        '''Install HubModule by name search result'''
        result = module_server.search_module(name=name, version=version)
        for item in result:
            if name.lower() == item['name'].lower() and utils.Version(item['version']).match(version):
                return self._install_from_url(item['url'])

        module_infos = module_server.get_module_compat_info(name=name)
        # The HubModule with the specified name cannot be found
        if not module_infos:
            raise HubModuleNotFoundError(name=name, version=version)

        valid_infos = {}
        if version:
            for _ver, _info in module_infos.items():
                if utils.Version(_ver).match(version):
                    valid_infos[_ver] = _info
        else:
            valid_infos = module_infos.copy()

        # Cannot find a HubModule that meets the version
        if valid_infos:
            if not ignore_env_mismatch:
                raise EnvironmentMismatchError(name=name, info=valid_infos, version=version)

            # If `ignore_env_mismatch` is set, ignore the problem of environmental mismatch, such as PaddlePaddle or PaddleHub
            # version incompatibility. This may cause some unexpected problems during installation or running, but it is useful
            # in some cases, for example, the development version of PaddlePaddle(with version number `0.0.0`) is installed
            # locally.
            if version:
                if version in valid_infos:
                    url = valid_infos[version]['url']
                else:
                    raise HubModuleNotFoundError(name=name, info=module_infos, version=version)
            else:
                # Get the maximum version number.
                version = sorted([utils.Version(_v) for _v in valid_infos.keys()])[-1]
                url = valid_infos[str(version)]['url']
            log.logger.warning('Ignore environmental mismatch of The Module {}-{}'.format(name, version))
            return self._install_from_url(url)
        raise HubModuleNotFoundError(name=name, info=module_infos, version=version)

    def _install_from_source(self, name: str, version: str, source: str, update: bool = False,
                             branch: str = None) -> HubModule:
        '''Install a HubModule from git repository'''
        result = module_server.search_module(name=name, source=source, version=version, update=update, branch=branch)
        for item in result:
            if item['name'] == name and item['version'].match(version):

                # uninstall local module
                local_module = self.search(name)
                if local_module and local_module.source == source and local_module.branch == branch:
                    self._local_modules[name] = local_module
                    return self._local_modules[name]

                if os.path.exists(self._get_normalized_path(name)):
                    self.uninstall(name)

                installed_path = self._get_normalized_path(name)
                shutil.copytree(item['path'], installed_path)

                source_info_file = os.path.join(installed_path, '_source_info.yaml')
                with open(source_info_file, 'w') as file:
                    file.write('source: {}\n'.format(source))
                    file.write('branch: {}'.format(branch))

                # Install python package requirements, This behavior needs to occur before Module.load,
                # otherwise the model will fail to load due to missing dependencies.
                self._install_module_requirements(installed_path)

                self._local_modules[name] = HubModule.load(installed_path)

                if version:
                    log.logger.info('Successfully installed {}-{}'.format(name, version))
                else:
                    log.logger.info('Successfully installed {}'.format(name))
                return self._local_modules[name]

        raise HubModuleNotFoundError(name=name, version=version, source=source)

    def _install_from_directory(self, directory: str) -> HubModule:
        '''Install a HubModule from directory containing module.py'''
        module_info = HubModule.load_module_info(directory)

        # A temporary directory is copied here for two purposes:
        # 1. Avoid affecting user-specified directory (for example, a __pycache__
        #    directory will be generated).
        # 2. HubModule is essentially a python package. When internal package
        #    references are made in it, the correct package name is required.
        with utils.generate_tempdir() as _dir:
            tempdir = os.path.join(_dir, module_info.name)
            tempdir = self._get_normalized_name(tempdir)
            shutil.copytree(directory, tempdir)

            # Uninstall local module
            if os.path.exists(self._get_normalized_path(module_info.name)):
                self.uninstall(module_info.name)

            shutil.copytree(directory, self._get_normalized_path(module_info.name))

            # Install python package requirements, This behavior needs to occur before Module.load,
            # otherwise the model will fail to load due to missing dependencies.
            self._install_module_requirements(directory)

            hub_module_cls = HubModule.load(self._get_normalized_path(module_info.name))
            self._local_modules[module_info.name] = hub_module_cls

            log.logger.info('Successfully installed {}-{}'.format(hub_module_cls.name, hub_module_cls.version))
            return hub_module_cls

    def _install_from_archive(self, archive: str) -> HubModule:
        '''Install HubModule from archive file (eg xxx.tar.gz)'''
        with utils.generate_tempdir() as _tdir:
            with log.ProgressBar('Decompress {}'.format(archive)) as bar:
                for path, ds, ts in xarfile.unarchive_with_progress(archive, _tdir):
                    bar.update(float(ds) / ts)

            # Sometimes the path contains '.'
            path = os.path.normpath(path)
            directory = os.path.join(_tdir, path.split(os.sep)[0])
            return self._install_from_directory(directory)

    def _install_module_requirements(self, directory: str):

        rfile = os.path.join(directory, 'requirements.txt')
        if not os.path.exists(rfile):
            return

        file = utils.get_record_file()
        with open(file, 'a') as _stream:

            with log.logger.processing('Installing dependent packages from {}'.format(rfile)):
                result = pypi.install_from_file(rfile, ostream=_stream, estream=_stream)
                if result:
                    log.logger.info('Successfully installed dependent packages.')
                else:
                    log.logger.warning(
                        'Some errors occurred while installing dependent packages. Detailed error information can be found in the {}.'
                        .format(file))

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

import argparse
import os
import platform
import json
import multiprocessing
import time
import signal

import paddlehub as hub
from paddlehub.commands import register
from paddlehub.serving import app_compat as app
from paddlehub.env import CONF_HOME
from paddlehub.serving.http_server import run_all, StandaloneApplication
from paddlehub.utils import log
from paddlehub.utils.utils import is_port_occupied
from paddlehub.server.server import CacheUpdater


def number_of_workers():
    '''
    Get suitable quantity of workers based on empirical formula.
    '''
    return (multiprocessing.cpu_count() * 2) + 1


def pid_is_exist(pid: int):
    '''
    Try to kill process by PID.

    Args:
        pid(int): PID of process to be killed.

    Returns:
         True if PID will be killed.

    Examples:
    .. code-block:: python

        pid_is_exist(pid=8866)
    '''
    try:
        os.kill(pid, 0)
    except:
        return False
    else:
        return True


@register(name='hub.serving', description='Start Module Serving or Bert Service for online predicting.')
class ServingCommand:
    name = "serving"
    module_list = []

    def dump_pid_file(self):
        '''
        Write PID info to file.
        '''
        pid = os.getpid()
        filepath = os.path.join(CONF_HOME, "serving_" + str(self.args.port) + ".json")
        if os.path.exists(filepath):
            os.remove(filepath)
        with open(filepath, "w") as fp:
            info = {"pid": pid, "module": self.args.modules, "start_time": time.time()}
            json.dump(info, fp)

    @staticmethod
    def load_pid_file(filepath: str, port: int = None):
        '''
        Read PID info from file.
        '''
        if port is None:
            port = os.path.basename(filepath).split(".")[0].split("_")[1]
        if not os.path.exists(filepath):
            log.logger.error(
                "PaddleHub Serving config file is not exists, please confirm the port [%s] you specified is correct." %
                port)
            return False
        with open(filepath, "r") as fp:
            info = json.load(fp)
            return info

    def stop_serving(self, port: int):
        '''
        Stop PaddleHub-Serving by port.
        '''
        filepath = os.path.join(CONF_HOME, "serving_" + str(port) + ".json")
        info = self.load_pid_file(filepath, port)
        if info is False:
            return
        pid = info["pid"]
        module = info["module"]
        start_time = info["start_time"]
        CacheUpdater("hub_serving_stop", module=module, addition={"period_time": time.time() - start_time}).start()
        if os.path.exists(filepath):
            os.remove(filepath)

        if not pid_is_exist(pid):
            log.logger.info("PaddleHub Serving has been stopped.")
            return
        log.logger.info("PaddleHub Serving will stop.")
        if platform.system() == "Windows":
            os.kill(pid, signal.SIGTERM)
        else:
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                os.kill(pid, signal.SIGTERM)

    @staticmethod
    def start_bert_serving(args):
        '''
        Start bert serving server.
        '''
        if platform.system() != "Linux":
            log.logger.error("Error. Bert Service only support linux.")
            return False

        if is_port_occupied("127.0.0.1", args.port) is True:
            log.logger.error("Port %s is occupied, please change it." % args.port)
            return False

        from paddle_gpu_serving.run import BertServer
        bs = BertServer(with_gpu=args.use_gpu)
        bs.with_model(model_name=args.modules[0])
        CacheUpdater("hub_bert_service", module=args.modules[0], version="0.0.0").start()
        bs.run(gpu_index=args.gpu, port=int(args.port))

    def preinstall_modules(self):
        '''
        Install module by PaddleHub and get info of this module.
        '''
        for key, value in self.modules_info.items():
            init_args = value["init_args"]
            CacheUpdater("hub_serving_start", module=key, version=init_args.get("version", "0.0.0")).start()
            if "directory" not in init_args:
                init_args.update({"name": key})
            m = hub.Module(**init_args)
            method_name = m.serving_func_name
            if method_name is None:
                raise RuntimeError("{} cannot be use for " "predicting".format(key))
                exit(1)
            serving_method = getattr(m, method_name)
            category = str(m.type).split("/")[0].upper()
            self.modules_info[key].update({
                "method_name": method_name,
                "version": m.version,
                "category": category,
                "module": m,
                "name": m.name,
                "serving_method": serving_method
            })

    def start_app_with_args(self):
        '''
        Start one PaddleHub-Serving instance by arguments with gunicorn.
        '''
        module = self.modules_info
        if module is not None:
            port = self.args.port
            if is_port_occupied("127.0.0.1", port) is True:
                log.logger.error("Port %s is occupied, please change it." % port)
                return False
            self.preinstall_modules()
            options = {"bind": "0.0.0.0:%s" % port, "workers": self.args.workers}
            self.dump_pid_file()
            StandaloneApplication(app.create_app(init_flag=False, configs=self.modules_info), options).run()
        else:
            log.logger.error("Lack of necessary parameters!")

    def start_zmq_serving_with_args(self):
        '''
        Start one PaddleHub-Serving instance by arguments with zmq.
        '''
        if self.modules_info is not None:
            for module, info in self.modules_info.items():
                CacheUpdater("hub_serving_start", module=module, version=info['init_args']['version']).start()
            front_port = self.args.port
            if is_port_occupied("127.0.0.1", front_port) is True:
                log.logger.error("Port %s is occupied, please change it." % front_port)
                return False
            back_port = int(front_port) + 1
            for index in range(100):
                if not is_port_occupied("127.0.0.1", back_port):
                    break
                else:
                    back_port = int(back_port) + 1
            else:
                raise RuntimeError(
                    "Port from %s to %s is occupied, please use another port" % (int(front_port) + 1, back_port))
            self.dump_pid_file()
            run_all(self.modules_info, self.args.gpu, front_port, back_port)

        else:
            log.logger.error("Lack of necessary parameters!")

    def start_single_app_with_args(self):
        '''
        Start one PaddleHub-Serving instance by arguments with flask.
        '''
        module = self.modules_info
        if module is not None:
            port = self.args.port
            if is_port_occupied("127.0.0.1", port) is True:
                log.logger.error("Port %s is occupied, please change it." % port)
                return False
            self.preinstall_modules()
            self.dump_pid_file()
            app.run(configs=self.modules_info, port=port)
        else:
            log.logger.error("Lack of necessary parameters!")

    def start_serving(self):
        '''
        Start PaddleHub-Serving with flask and gunicorn
        '''
        if self.args.use_gpu:
            if self.args.use_multiprocess:
                log.logger.warning('`use_multiprocess` will be ignored if specify `use_gpu`.')
            self.start_zmq_serving_with_args()
        else:
            if self.args.use_multiprocess:
                if platform.system() == "Windows":
                    log.logger.warning(
                        "Warning: Windows cannot use multiprocess working mode, PaddleHub Serving will switch to single process mode"
                    )
                    self.start_single_app_with_args()
                else:
                    self.start_app_with_args()
            else:
                self.start_single_app_with_args()

    @staticmethod
    def show_help():
        str = "serving <option>\n"
        str += "\tManage PaddleHub Serving.\n"
        str += "sub command:\n"
        str += "1. start\n"
        str += "\tStart PaddleHub Serving.\n"
        str += "2. stop\n"
        str += "\tStop PaddleHub Serving.\n"
        str += "3. start bert_service\n"
        str += "\tStart Bert Service.\n"
        str += "\n"
        str += "[start] option:\n"
        str += "--modules/-m [module1==version, module2==version...]\n"
        str += "\tPre-install modules via the parameter list.\n"
        str += "--port/-p XXXX\n"
        str += "\tUse port XXXX for serving.\n"
        str += "--use_multiprocess\n"
        str += "\tChoose multoprocess mode, cannot be use on Windows.\n"
        str += "--modules_info\n"
        str += "\tSet module config in PaddleHub Serving."
        str += "--config/-c file_path\n"
        str += "\tUse configs in file to start PaddleHub Serving. "
        str += "Other parameters will be ignored if you specify the parameter.\n"
        str += "\n"
        str += "[stop] option:\n"
        str += "--port/-p XXXX\n"
        str += "\tStop PaddleHub Serving on port XXXX safely.\n"
        str += "\n"
        str += "[start bert_service] option:\n"
        str += "--modules/-m\n"
        str += "\tPre-install modules via the parameter.\n"
        str += "--port/-p XXXX\n"
        str += "\tUse port XXXX for serving.\n"
        str += "--use_gpu\n"
        str += "\tUse gpu for predicting if specifies the parameter.\n"
        str += "--gpu\n"
        str += "\tSpecify the GPU devices to use.\n"
        print(str)

    def parse_args(self):
        if self.args.config is not None:
            if os.path.exists(self.args.config):
                with open(self.args.config, "r") as fp:
                    # self.args.config = json.load(fp)
                    self.args_config = json.load(fp)
                self.args.use_gpu = self.args_config.get('use_gpu', False)
                self.args.use_multiprocess = self.args_config.get('use_multiprocess', False)
                self.modules_info = self.args_config["modules_info"]
                self.args.port = self.args_config.get('port', 8866)
                if self.args.use_gpu:
                    self.args.gpu = self.args_config.get('gpu', '0')
                else:
                    self.args.gpu = self.args_config.get('gpu', None)
                self.args.use_gpu = self.args_config.get('use_gpu', False)
                if self.args.use_multiprocess:
                    self.args.workers = self.args_config.get('workers', number_of_workers())
                else:
                    self.args.workers = self.args_config.get('workers', None)
            else:
                raise RuntimeError("{} not exists.".format(self.args.config))
                exit(1)
        else:
            self.modules_info = {}
            for item in self.args.modules:
                version = None
                if "==" in item:
                    module = item.split("==")[0]
                    version = item.split("==")[1]
                else:
                    module = item
                self.modules_info.update({module: {"init_args": {"version": version}, "predict_args": {}}})
        if self.args.gpu:
            self.args.gpu = self.args.gpu.split(',')

        return self.modules_info

    def execute(self, argv):
        self.show_in_help = True
        self.description = "Start Module Serving or Bert Service for online predicting."
        self.parser = argparse.ArgumentParser(
            description=self.__class__.__doc__, prog='hub serving', usage='%(prog)s', add_help=True)
        self.parser.add_argument("command")
        self.parser.add_argument("sub_command")
        self.parser.add_argument("bert_service", nargs="?")
        self.sub_parse = self.parser.add_mutually_exclusive_group(required=False)
        self.parser.add_argument("--use_gpu", action="store_true", default=False)
        self.parser.add_argument("--use_multiprocess", action="store_true", default=False)
        self.parser.add_argument("--modules", "-m", nargs="+")
        self.parser.add_argument("--config", "-c", nargs="?")
        self.parser.add_argument("--port", "-p", nargs="?", default=8866)
        self.parser.add_argument("--gpu", "-i", nargs="?", default='0')
        self.parser.add_argument("--use_singleprocess", action="store_true", default=False)
        self.parser.add_argument("--modules_info", "-mi", default={}, type=json.loads)
        self.parser.add_argument("--workers", "-w", nargs="?", default=number_of_workers())
        try:
            self.args = self.parser.parse_args()
        except:
            ServingCommand.show_help()
            return False
        if self.args.sub_command == "start":
            if self.args.bert_service == "bert_service":
                ServingCommand.start_bert_serving(self.args)
            else:
                self.parse_args()
                self.start_serving()
        elif self.args.sub_command == "stop":
            if self.args.bert_service == "bert_service":
                log.logger.warning("Please stop Bert Service by kill process by yourself")
            elif self.args.bert_service is None:
                self.stop_serving(port=self.args.port)
        else:
            ServingCommand.show_help()

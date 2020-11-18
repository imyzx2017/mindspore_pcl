# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""kernel build server for ascend"""
import sys
from mindspore._extends.remote.kernel_build_server import Messager, get_logger
from mindspore._extends.parallel_compile.tbe_compiler.tbe_process import create_tbe_parallel_process, op_select_format, check_supported
from mindspore._extends.parallel_compile.akg_compiler.akg_process import create_akg_parallel_process

class TbeBuilder:
    """Tbe building wrapper"""

    def __init__(self):
        self.tbe_builder = create_tbe_parallel_process()

    def start(self, json):
        return self.tbe_builder.start_compile_op(json)

    def wait(self):
        return self.tbe_builder.wait_one()

    def reset(self):
        self.tbe_builder.reset_task_info()

    def exit(self):
        self.tbe_builder.exit()

class AkgBuilder:
    """Akg building wrapper"""

    def __init__(self):
        pass

    def create(self, process_num, waitime):
        self.akg_builder = create_akg_parallel_process(process_num, waitime)

    def accept_json(self, json):
        return self.akg_builder.accept_json(json)

    def compile(self):
        return self.akg_builder.compile()

class AscendMessager(Messager):
    '''
    Ascend Messager
    It works as a server, communicating with c++ client.
    '''

    def __init__(self, fdin, fdout):
        super().__init__(fdin, fdout)
        get_logger().info('[TRACE]', 'Ascend Messager init...')
        self.tbe_builder = TbeBuilder()
        self.akg_builder = AkgBuilder()

    def handle(self):
        """
        Communicate with remote client.
        Reference protocol between them at PR#3821 and PR#3935
        """
        arg = self.get_message()
        if arg == 'TBE/START':
            self.send_ack()
            json = self.get_message()
            res = self.tbe_builder.start(json)
            self.send_res(res)
        elif arg == 'TBE/WAIT':
            self.send_ack()
            task_id, res, pre = self.tbe_builder.wait()
            get_logger().debug('[TRACE]', str(task_id) + '/' + str(res) + '/' + str(pre))
            if self.get_message() != 'CONTINUE':
                self.send_ack(False)
                self.exit()
            self.send_res(task_id)
            if self.get_message() != 'CONTINUE':
                self.send_ack(False)
                self.exit()
            self.send_res(res)
            if self.get_message() != 'CONTINUE':
                self.send_ack(False)
                self.exit()
            self.send_res(pre)
        elif arg == 'TBE/RESET':
            self.tbe_builder.reset()
            self.send_ack()
        elif arg == 'AKG/START':
            self.send_ack()
            process_num_str = self.get_message()
            self.send_ack()
            wait_time_str = self.get_message()
            self.akg_builder.create(int(process_num_str), int(wait_time_str))
            self.send_ack()
        elif arg == 'AKG/DATA':
            self.send_ack()
            while True:
                req = self.get_message()
                if req.startswith('{'):
                    self.akg_builder.accept_json(req)
                    self.send_ack()
                elif req == 'AKG/WAIT':
                    res = self.akg_builder.compile()
                    self.send_res(res)
                    break
                else:
                    self.send_ack(False)
                    break
        elif arg == 'FORMAT':
            self.send_ack()
            json = self.get_message()
            self.send_res(op_select_format(json))
        elif arg == 'SUPPORT':
            self.send_ack()
            json = self.get_message()
            get_logger().debug('[SUPPORT]', json)
            try:
                res = check_supported(json)
            except json.decoder.JSONDecodeError:
                self.send_ack(False)
                self.exit()
            finally:
                pass
            self.send_res(res)
        else:
            self.send_ack(False)
            self.exit()

    def exit(self):
        self.tbe_builder.reset()
        self.tbe_builder.exit()
        get_logger().info('[TRACE]', 'Ascend Messager Exit...')
        exit()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise Exception('Incorrect argv: {}'.format(sys.argv))
    get_logger().debug('[TRACE]', 'argv: ' + str(sys.argv))
    messager = AscendMessager(int(sys.argv[1]), int(sys.argv[2]))
    messager.run()

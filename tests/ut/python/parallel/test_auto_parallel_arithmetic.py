# Copyright 2019 Huawei Technologies Co., Ltd
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

import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import _executor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.parallel._utils import _reset_op_id as reset_op_id
from tests.ut.python.ops.test_math_ops import VirtualLoss

context.set_context(mode=context.GRAPH_MODE)


grad_all = C.GradOperation(get_all=True)


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x, y, b):
        predict = self.network(x, y, b)
        return self.loss(predict)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x, y, b):
        return grad_all(self.network)(x, y, b)


def compile_net(net, x, y, b, phase):
    net.set_auto_parallel()
    net.set_train()
    _executor.compile(net, x, y, b, phase=phase)


def test_auto_parallel_arithmetic():
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.matmul = P.MatMul()
            self.floordiv = P.FloorDiv()

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.floordiv(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    net = NetWithLoss(Net())
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    reset_op_id()

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 128]), dtype=ms.float32)
    b = Tensor(np.ones([64, 128]), dtype=ms.float32)
    compile_net(net, x, y, b, phase='train')
    strategies = _executor._get_shard_strategy(net)
    expected_strategies = {'Default/network-Net/FloorDiv-op0': [[2, 4], [2, 4]],
                           'Default/network-Net/MatMul-op1': [[2, 1], [1, 4]]}
    assert strategies == expected_strategies


def test_auto_parallel_arithmetic_broadcast_both():
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.matmul = P.MatMul()
            self.floordiv = P.FloorDiv()

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.floordiv(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    net = NetWithLoss(Net())
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    reset_op_id()

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 1]), dtype=ms.float32)
    b = Tensor(np.ones([1, 64]), dtype=ms.float32)
    compile_net(net, x, y, b, phase='train')
    strategies = _executor._get_shard_strategy(net)
    expected_strategies = {'Default/network-Net/FloorDiv-op0': [[8, 1], [1, 1]],
                           'Default/network-Net/MatMul-op1': [[8, 1], [1, 1]]}
    assert strategies == expected_strategies


def test_auto_parallel_arithmetic_broadcast_right():
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.matmul = P.MatMul()
            self.floordiv = P.FloorDiv()

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.floordiv(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    net = NetWithLoss(Net())
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    reset_op_id()

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 32]), dtype=ms.float32)
    b = Tensor(np.ones([32]), dtype=ms.float32)
    compile_net(net, x, y, b, phase='train')
    strategies = _executor._get_shard_strategy(net)
    expected_strategies = {'Default/network-Net/FloorDiv-op0': [[4, 2], [2]],
                           'Default/network-Net/MatMul-op1': [[4, 1], [1, 2]]}
    assert strategies == expected_strategies


def test_auto_parallel_arithmetic_broadcast_left():
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.matmul = P.MatMul()
            self.floordiv = P.FloorDiv()

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.floordiv(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    net = NetWithLoss(Net())
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    reset_op_id()

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 32]), dtype=ms.float32)
    b = Tensor(np.ones([128, 64, 32]), dtype=ms.float32)
    compile_net(net, x, y, b, phase="train")
    strategies = _executor._get_shard_strategy(net)
    expected_strategies = {'Default/network-Net/FloorDiv-op0': [[4, 2], [1, 4, 2]],
                           'Default/network-Net/MatMul-op1': [[4, 1], [1, 2]]}
    assert strategies == expected_strategies

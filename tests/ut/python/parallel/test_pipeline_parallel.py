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
# ============================================================================
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import _executor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from tests.ut.python.ops.test_math_ops import VirtualLoss


grad_all = C.GradOperation(get_all=True)


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x, y):
        predict = self.network(x, y)
        return self.loss(predict)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x, y):
        return grad_all(self.network)(x, y)


class Net(nn.Cell):
    def __init__(self, axis=0, stage1=0, stage2=0, strategy1=None, strategy2=None, shape=None, target=""):
        super().__init__()
        if shape is None:
            shape = [64, 64]
        self.gatherv2 = P.GatherV2().shard(strategy1).add_prim_attr("primitive_target", target)
        self.mul = P.Mul().shard(strategy2)
        self.index = Tensor(np.ones(shape), dtype=ms.int32)
        self.gatherv2.set_stage(stage1)
        self.mul.set_stage(stage2)
        self.axis = axis

    def construct(self, x, y):
        out = self.gatherv2(x, self.index, self.axis)
        out = self.mul(out, y)
        return out


def test_gatherv2_semi_samestage1():
    context.set_auto_parallel_context(device_num=8, global_rank=0, \
        parallel_mode="semi_auto_parallel", pipeline_stages=2)
    strategy1 = ((1, 2), (1, 1))
    strategy2 = ((2, 1, 1), (2, 1, 1))
    net = GradWrap(NetWithLoss(Net(0, 0, 0, strategy1, strategy2)))
    net.set_auto_parallel()

    x = Tensor(np.ones([64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    net.set_train()
    _executor.compile(net, x, y)

def test_gatherv2_semi_samestage2():
    context.set_auto_parallel_context(device_num=8, global_rank=5, \
        parallel_mode="semi_auto_parallel", pipeline_stages=2)
    strategy1 = ((1, 2), (1, 1))
    strategy2 = ((2, 1, 1), (2, 1, 1))
    net = GradWrap(NetWithLoss(Net(0, 1, 1, strategy1, strategy2)))
    net.set_auto_parallel()

    x = Tensor(np.ones([64, 64]), dtype=ms.float32)
    y = Tensor(np.ones([64, 64, 64]), dtype=ms.float32)
    net.set_train()
    _executor.compile(net, x, y)

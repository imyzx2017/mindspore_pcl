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
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class NetConv2d(nn.Cell):
    def __init__(self):
        super(NetConv2d, self).__init__()
        out_channel = 2
        kernel_size = 1
        self.conv = P.Conv2D(out_channel,
                             kernel_size,
                             mode=1,
                             pad_mode="valid",
                             pad=0,
                             stride=1,
                             dilation=1,
                             group=1)

    def construct(self, x, w):
        return self.conv(x, w)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_conv2d():
    x = Tensor(np.arange(1 * 3 * 3 * 3).reshape(1, 3, 3, 3).astype(np.float32))
    w = Tensor(np.arange(2 * 3 * 1 * 1).reshape(2, 3, 1, 1).astype(np.float32))
    expect = np.array([[[[45, 48, 51],
                         [54, 57, 60],
                         [63, 66, 69]],
                        [[126, 138, 150],
                         [162, 174, 186],
                         [198, 210, 222]]]]).astype(np.float32)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", max_device_memory="0.2GB")
    conv2d = NetConv2d()
    output = conv2d(x, w)
    assert (output.asnumpy() == expect).all()
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    conv2d = NetConv2d()
    output = conv2d(x, w)
    assert (output.asnumpy() == expect).all()

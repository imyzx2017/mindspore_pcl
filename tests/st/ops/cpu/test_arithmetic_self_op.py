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
import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class SquareNet(nn.Cell):
    def __init__(self):
        super(SquareNet, self).__init__()
        self.square = P.Square()

    def construct(self, x):
        return self.square(x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_square():
    x = np.array([1, 2, 3]).astype(np.int16)
    net = SquareNet()
    output = net(Tensor(x))
    expect_output = np.array([1, 4, 9]).astype(np.int16)
    print(output)
    assert np.all(output.asnumpy() == expect_output)

    x = np.array([1, 2, 3]).astype(np.int32)
    net = SquareNet()
    output = net(Tensor(x))
    expect_output = np.array([1, 4, 9]).astype(np.int32)
    print(output)
    assert np.all(output.asnumpy() == expect_output)

    x = np.array([1, 2, 3]).astype(np.int64)
    net = SquareNet()
    output = net(Tensor(x))
    expect_output = np.array([1, 4, 9]).astype(np.int64)
    print(output)
    assert np.all(output.asnumpy() == expect_output)

    x = np.array([1, 2, 3]).astype(np.float16)
    net = SquareNet()
    output = net(Tensor(x))
    expect_output = np.array([1, 4, 9]).astype(np.float16)
    print(output)
    assert np.all(output.asnumpy() == expect_output)

    x = np.array([1, 2, 3]).astype(np.float32)
    net = SquareNet()
    output = net(Tensor(x))
    expect_output = np.array([1, 4, 9]).astype(np.float32)
    print(output)
    assert np.all(output.asnumpy() == expect_output)

    x = np.array([1, 2, 3]).astype(np.float64)
    net = SquareNet()
    output = net(Tensor(x))
    expect_output = np.array([1, 4, 9]).astype(np.float64)
    print(output)
    assert np.all(output.asnumpy() == expect_output)

test_square()

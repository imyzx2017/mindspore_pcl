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
from mindspore.common.tensor import Tensor
from mindspore.nn import Cell
from mindspore.ops import operations as P


class Net(Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.lessequal = P.LessEqual()

    def construct(self, x, y):
        return self.lessequal(x, y)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_lessequal():
    x = Tensor(np.array([[1, 2, 3]]).astype(np.float32))
    y = Tensor(np.array([[2]]).astype(np.float32))
    expect = [[True, True, False]]
    x1 = Tensor(np.array([[1, 2, 3]]).astype(np.int16))
    y1 = Tensor(np.array([[2]]).astype(np.int16))
    expect = [[True, True, False]]
    x2 = Tensor(np.array([[1, 2, 3]]).astype(np.uint8))
    y2 = Tensor(np.array([[2]]).astype(np.uint8))
    expect = [[True, True, False]]

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    lessequal = Net()
    output = lessequal(x, y)
    assert np.all(output.asnumpy() == expect)
    output = lessequal(x1, y1)
    assert np.all(output.asnumpy() == expect)
    output = lessequal(x2, y2)
    assert np.all(output.asnumpy() == expect)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    lessequal = Net()
    output = lessequal(x, y)
    assert np.all(output.asnumpy() == expect)
    output = lessequal(x1, y1)
    assert np.all(output.asnumpy() == expect)
    output = lessequal(x2, y2)
    assert np.all(output.asnumpy() == expect)

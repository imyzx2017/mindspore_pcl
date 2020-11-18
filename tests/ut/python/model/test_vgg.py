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
"""test_vgg"""
import numpy as np
import pytest

from mindspore import Tensor
from model_zoo.official.cv.vgg16.src.vgg import vgg16
from model_zoo.official.cv.vgg16.src.config import cifar_cfg as cfg
from ..ut_filter import non_graph_engine


@non_graph_engine
def test_vgg16():
    inputs = Tensor(np.random.rand(1, 3, 112, 112).astype(np.float32))
    net = vgg16(args=cfg)
    with pytest.raises(ValueError):
        print(net.construct(inputs))

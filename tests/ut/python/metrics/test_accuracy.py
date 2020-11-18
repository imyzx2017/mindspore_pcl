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
"""test accuracy"""
import math
import numpy as np
import pytest

from mindspore import Tensor
from mindspore.nn.metrics import Accuracy


def test_classification_accuracy():
    """test_classification_accuracy"""
    x = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
    y = Tensor(np.array([1, 0, 1]))
    y2 = Tensor(np.array([[0, 1], [1, 0], [0, 1]]))
    metric = Accuracy('classification')
    metric.clear()
    metric.update(x, y)
    accuracy = metric.eval()
    accuracy2 = metric(x, y2)
    assert math.isclose(accuracy, 2 / 3)
    assert math.isclose(accuracy2, 2 / 3)


def test_multilabel_accuracy():
    x = Tensor(np.array([[0, 1, 0, 1], [1, 0, 1, 1], [0, 0, 0, 1]]))
    y = Tensor(np.array([[0, 1, 1, 1], [0, 1, 1, 1], [0, 0, 0, 1]]))
    metric = Accuracy('multilabel')
    metric.clear()
    metric.update(x, y)
    accuracy = metric.eval()
    assert accuracy == 1 / 3


def test_shape_accuracy():
    x = Tensor(np.array([[0, 1, 0, 1], [1, 0, 1, 1], [0, 0, 0, 1]]))
    y = Tensor(np.array([[0, 1, 1, 1], [0, 1, 1, 1]]))
    metric = Accuracy('multilabel')
    metric.clear()
    with pytest.raises(ValueError):
        metric.update(x, y)


def test_shape_accuracy2():
    x = Tensor(np.array([[0, 1, 0, 1], [1, 0, 1, 1], [0, 0, 0, 1]]))
    y = Tensor(np.array([0, 1, 1, 1]))
    metric = Accuracy('multilabel')
    metric.clear()
    with pytest.raises(ValueError):
        metric.update(x, y)


def test_shape_accuracy3():
    x = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
    y = Tensor(np.array([[1, 0, 1], [1, 1, 1]]))
    metric = Accuracy('classification')
    metric.clear()
    with pytest.raises(ValueError):
        metric.update(x, y)


def test_shape_accuracy4():
    x = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
    y = Tensor(np.array(1))
    metric = Accuracy('classification')
    metric.clear()
    with pytest.raises(ValueError):
        metric.update(x, y)


def test_type_accuracy():
    with pytest.raises(TypeError):
        Accuracy('test')

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
"""Utils for MindExplain"""

__all__ = [
    'ForwardProbe',
    'calc_auc',
    'calc_correlation',
    'format_tensor_to_ndarray',
    'generate_one_hot',
    'rank_pixels',
    'resize',
    'retrieve_layer_by_name',
    'retrieve_layer',
    'unify_inputs',
    'unify_targets'
]

from typing import Tuple, Union
import math

import numpy as np
from PIL import Image

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops.operations as op

_Array = np.ndarray
_Module = nn.Cell
_Tensor = ms.Tensor


def generate_one_hot(indices, depth):
    r"""
    Simple wrap of OneHot operation, the on_value an off_value are fixed to 1.0
    and 0.0.
    """
    on_value = ms.Tensor(1.0, ms.float32)
    off_value = ms.Tensor(0.0, ms.float32)
    weights = op.OneHot()(indices, depth, on_value, off_value)
    return weights


def unify_inputs(inputs) -> tuple:
    """Unify inputs of explainer."""
    if isinstance(inputs, tuple):
        return inputs
    if isinstance(inputs, ms.Tensor):
        inputs = (inputs,)
    elif isinstance(inputs, np.ndarray):
        inputs = (ms.Tensor(inputs),)
    else:
        raise TypeError(
            'inputs must be one of [tuple, ms.Tensor or np.ndarray], '
            'but get {}'.format(type(inputs)))
    return inputs


def unify_targets(targets) -> ms.Tensor:
    """Unify targets labels of explainer."""
    if isinstance(targets, ms.Tensor):
        return targets
    if isinstance(targets, list):
        targets = ms.Tensor(targets, dtype=ms.int32)
    if isinstance(targets, int):
        targets = ms.Tensor([targets], dtype=ms.int32)
    else:
        raise TypeError(
            'targets must be one of [int, list or ms.Tensor], '
            'but get {}'.format(type(targets)))
    return targets


def retrieve_layer_by_name(model: _Module, layer_name: str):
    """
    Retrieve the layer in the model by the given layer_name.

    Args:
        model (_Module): model which contains the target layer
        layer_name (str): name of target layer

    Return:
        - target_layer (_Module)

    Raise:
        ValueError: is module with given layer_name is not found in the model,
            raise ValueError.

    """
    if not isinstance(layer_name, str):
        raise TypeError('layer_name should be type of str, but receive {}.'
                        .format(type(layer_name)))

    if not layer_name:
        return model

    target_layer = None
    for name, cell in model.cells_and_names():
        if name == layer_name:
            target_layer = cell
            return target_layer

    if target_layer is None:
        raise ValueError(
            'Cannot match {}, please provide target layer'
            'in the given model.'.format(layer_name))
    return None


def retrieve_layer(model: _Module, target_layer: Union[str, _Module] = ''):
    """
    Retrieve the layer in the model.

    'target' can be either a layer name or a Cell object. Given the layer name,
    the method will search thourgh the model and return the matched layer. If a
    Cell object is provided, it will check whether the given layer exists
    in the model. If target layer is not found in the model, ValueError will
    be raised.

    Args:
        model (_Module): the model to retrieve the target layer
        target_layer (Union[str, _Module]): target layer to retrieve. Can be
        either string (layer name) or the Cell object. If '' is provided,
        the input model will be returned.

    Return:
        target layer (_Module)
    """
    if isinstance(target_layer, str):
        target_layer = retrieve_layer_by_name(model, target_layer)
        return target_layer

    if isinstance(target_layer, _Module):
        for _, cell in model.cells_and_names():
            if target_layer is cell:
                return target_layer
        raise ValueError(
            'Model not contain cell {}, fail to probe.'.format(target_layer)
        )
    raise TypeError('layer_name must have type of str or ms.nn.Cell,'
                    'but receive {}'.format(type(target_layer)))


class ForwardProbe:
    """
    Probe to capture output of specific layer in a given model.

    Args:
        target_layer (_Module): name of target layer or just provide the
        target layer.

    """

    def __init__(self, target_layer: _Module):
        self._target_layer = target_layer
        self._original_construct = self._target_layer.construct
        self._intermediate_tensor = None

    @property
    def value(self):
        return self._intermediate_tensor

    def __enter__(self):
        self._target_layer.construct = self._new_construct
        return self

    def __exit__(self, *_):
        self._target_layer.construct = self._original_construct
        self._intermediate_tensor = None
        return False

    def _new_construct(self, *inputs):
        outputs = self._original_construct(*inputs)
        self._intermediate_tensor = outputs
        return outputs


def format_tensor_to_ndarray(x: Union[ms.Tensor, np.ndarray]) -> np.ndarray:
    """Unify `mindspore.Tensor` and `np.ndarray` to `np.ndarray`. """
    if isinstance(x, ms.Tensor):
        x = x.asnumpy()

    if not isinstance(x, np.ndarray):
        raise TypeError('input should be one of [ms.Tensor or np.ndarray],'
                        ' but receive {}'.format(type(x)))
    return x


def calc_correlation(x: Union[ms.Tensor, np.ndarray],
                     y: Union[ms.Tensor, np.ndarray]) -> float:
    """Calculate Pearson correlation coefficient between two arrays. """
    x = format_tensor_to_ndarray(x)
    y = format_tensor_to_ndarray(y)
    faithfulness = -np.corrcoef(x, y)[0, 1]
    if math.isnan(faithfulness):
        return np.float(0)
    return faithfulness


def calc_auc(x: _Array) -> float:
    """Calculate the Aera under Curve."""
    # take mean for multiple patches if the model is fully convolutional model
    if len(x.shape) == 4:
        x = np.mean(np.mean(x, axis=2), axis=3)

    auc = (x.sum() - x[0] - x[-1]) / len(x)
    return float(auc)


def rank_pixels(inputs: _Array, descending: bool = True) -> _Array:
    """
    Generate rank order fo every pixel in an 2D array.

    The rank order start from 0 to (num_pixel-1). If descending is True, the
    rank order will generate in a descending order, otherwise in ascending
    order.

    Example:
        x = np.array([[4., 3., 1.], [5., 9., 1.]])
        rank_pixels(x, descending=True)
        >> np.array([[2, 3, 4], [1, 0, 5]])
        rank_pixels(x, descending=False)
        >> np.array([[3, 2, 0], [4, 5, 1]])
    """
    if len(inputs.shape) != 2:
        raise ValueError('Only support 2D array currently')
    flatten_saliency = inputs.reshape(-1)
    factor = -1 if descending else 1
    sorted_arg = np.argsort(factor * flatten_saliency, axis=0)
    flatten_rank = np.zeros_like(sorted_arg)
    flatten_rank[sorted_arg] = np.arange(0, sorted_arg.shape[0])
    rank_map = flatten_rank.reshape(inputs.shape)
    return rank_map


def resize(inputs: _Tensor, size: Tuple[int, int], mode: str) -> _Tensor:
    """
    Resize the intermediate layer _attribution to the same size as inputs.

    Args:
        inputs (ms.Tensor): the input tensor to be resized
        size (tupleint]): the targeted size resize to
        mode (str): the resize mode. Options: 'nearest_neighbor', 'bilinear'

    Returns:
        outputs (ms.Tensor): the resized tensor.

    Raises:
        ValueError: the resize mode is not in ['nearest_neighbor',
         'bilinear'].
    """
    h, w = size
    if mode == 'nearest_neighbor':
        resize_nn = op.ResizeNearestNeighbor((h, w))
        outputs = resize_nn(inputs)

    elif mode == 'bilinear':
        inputs_np = inputs.asnumpy()
        inputs_np = np.transpose(inputs_np, [0, 2, 3, 1])
        array_lst = []
        for inp in inputs_np:
            array = (np.repeat(inp, 3, axis=2) * 255).astype(np.uint8)
            image = Image.fromarray(array)
            image = image.resize(size, resample=Image.BILINEAR)
            array = np.asarray(image).astype(np.float32) / 255
            array_lst.append(array[:, :, 0:1])

        resized_np = np.transpose(array_lst, [0, 3, 1, 2])
        outputs = ms.Tensor(resized_np, inputs.dtype)
    else:
        raise ValueError('Unsupported resize mode {}'.format(mode))

    return outputs

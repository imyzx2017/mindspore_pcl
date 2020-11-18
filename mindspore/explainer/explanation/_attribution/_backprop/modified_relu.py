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
"""Explainer with modified ReLU."""

import mindspore.nn as nn
import mindspore.ops.operations as op

from .gradient import Gradient
from ...._utils import (
    unify_inputs,
    unify_targets,
)


class ModifiedReLU(Gradient):
    """Basic class for modified ReLU explanation."""

    def __init__(self, network, use_relu_backprop=False):
        super(ModifiedReLU, self).__init__(network)
        self.use_relu_backprop = use_relu_backprop
        self.hooked_list = []

    def __call__(self, inputs, targets):
        """
        Call function for `ModifiedReLU`, inherited by "Deconvolution" and "GuidedBackprop".

        Args:
            inputs (Tensor): The input data to be explained, a 4D tensor of shape :math:`(N, C, H, W)`.
            targets (Tensor, int): The label of interest. It should be a 1D or 0D tensor, or an integer.
                If it is a 1D tensor, its length should be the same as `inputs`.

        Returns:
            Tensor, a 4D tensor of shape :math:`(N, 1, H, W)`.

        Examples:
            >>> inputs = ms.Tensor(np.random.rand([1, 3, 224, 224]), ms.float32)
            >>> label = 5
            >>> # explainer is a "Deconvolution" or "GuidedBackprop" object, parse data and the target label to be
            >>> # explained and get the attribution
            >>> saliency = explainer(inputs, label)
        """

        self._verify_data(inputs, targets)
        inputs = unify_inputs(inputs)
        targets = unify_targets(targets)

        self._hook_relu_backward()
        gradients = self._grad_op(self._backward_model, inputs, targets)
        saliency = self._aggregation_fn(gradients)

        return saliency

    def _hook_relu_backward(self):
        """Set backward hook for ReLU layers."""
        for _, cell in self._backward_model.cells_and_names():
            if isinstance(cell, nn.ReLU):
                cell.register_backward_hook(self._backward_hook)
                self.hooked_list.append(cell)

    def _backward_hook(self, _, grad_inputs, grad_outputs):
        """Hook function for ReLU layers."""
        inputs = grad_inputs if self.use_relu_backprop else grad_outputs
        relu = op.ReLU()
        if isinstance(inputs, tuple):
            return relu(*inputs)
        return relu(inputs)


class Deconvolution(ModifiedReLU):
    """
    Deconvolution explanation.

    Deconvolution method is a modified version of Gradient method. For the original ReLU operation in the network to be
    explained, Deconvolution modifies the propagation rule from directly backpropagating gradients to backprpagating
    positive gradients.

    Note:
        The parsed `network` will be set to eval mode through `network.set_grad(False)` and `network.set_train(False)`.
        If you want to train the `network` afterwards, please reset it back to training mode through the opposite
        operations. To use `Deconvolution`, the `ReLU` operations in the network must be implemented with
        `mindspore.nn.Cell` object rather than `mindspore.ops.Operations.ReLU`. Otherwise, the results will not be
        correct.

    Args:
        network (Cell): The black-box model to be explained.

    Examples:
        >>> from mindspore.explainer.explanation import Deconvolution
        >>> net = resnet50(10)
        >>> param_dict = load_checkpoint("resnet50.ckpt")
        >>> load_param_into_net(net, param_dict)
        >>> # init Gradient with a trained network.
        >>> deconvolution = Deconvolution(net)
        >>> # parse data and the target label to be explained and get the saliency map
        >>> inputs = ms.Tensor(np.random.rand([1, 3, 224, 224]), ms.float32)
        >>> label = 5
        >>> saliency = deconvolution(inputs, label)
    """

    def __init__(self, network):
        super(Deconvolution, self).__init__(network, use_relu_backprop=True)


class GuidedBackprop(ModifiedReLU):
    """
    Guided-Backpropagation explanation.

    Guided-Backpropagation method is an extension of Gradient method. On top of the original ReLU operation in the
    network to be explained, Guided-Backpropagation introduces another ReLU operation to filter out the negative
    gradients during backpropagation.

    Note:
        The parsed `network` will be set to eval mode through `network.set_grad(False)` and `network.set_train(False)`.
        If you want to train the `network` afterwards, please reset it back to training mode through the opposite
        operations. To use `GuidedBackprop`, the `ReLU` operations in the network must be implemented with
        `mindspore.nn.Cell` object rather than `mindspore.ops.Operations.ReLU`. Otherwise, the results will not be
        correct.

    Args:
        network (Cell): The black-box model to be explained.

    Examples:
        >>> from mindspore.explainer.explanation import GuidedBackprop
        >>> net = resnet50(10)
        >>> param_dict = load_checkpoint("resnet50.ckpt")
        >>> load_param_into_net(net, param_dict)
        >>> # init Gradient with a trained network.
        >>> gbp = GuidedBackprop(net)
        >>> # parse data and the target label to be explained and get the saliency map
        >>> inputs = ms.Tensor(np.random.rand([1, 3, 224, 224]), ms.float32)
        >>> label = 5
        >>> saliency = gbp(inputs, label)
    """

    def __init__(self, network):
        super(GuidedBackprop, self).__init__(network, use_relu_backprop=False)

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
"""Gradient explainer."""
from copy import deepcopy

from mindspore import nn
from mindspore.ops import operations as op
from mindspore.train._utils import check_value_type
from ...._operators import reshape, sqrt, Tensor
from .._attribution import Attribution
from .backprop_utils import compute_gradients
from ...._utils import unify_inputs, unify_targets


def _get_hook(bntype, cache):
    """Provide backward hook function for BatchNorm layer in eval mode."""
    var, gamma, eps = cache
    if bntype == "2d":
        var = reshape(var, (1, -1, 1, 1))
        gamma = reshape(gamma, (1, -1, 1, 1))
    elif bntype == "1d":
        var = reshape(var, (1, -1, 1))
        gamma = reshape(gamma, (1, -1, 1))

    def reset_gradient(_, grad_input, grad_output):
        grad_output = grad_input[0] * gamma / sqrt(var + eps)
        return grad_output

    return reset_gradient


def _abs_max(gradients):
    """
    Transform gradients to saliency through abs then take max along
    channels.
    """
    gradients = op.Abs()(gradients)
    saliency = op.ReduceMax(keep_dims=True)(gradients, axis=1)
    return saliency


class Gradient(Attribution):
    r"""
    Provides Gradient explanation method.

    Gradient is the simplest attribution method which uses the naive gradients of outputs w.r.t inputs as the
    explanation.

    .. math::

        attribution = \frac{\partial{y}}{\partial{x}}

    Note:
        The parsed `network` will be set to eval mode through `network.set_grad(False)` and `network.set_train(False)`.
        If you want to train the `network` afterwards, please reset it back to training mode through the opposite
        operations.

    Args:
        network (Cell): The black-box model to be explained.

    Examples:
        >>> from mindspore.explainer.explanation import Gradient
        >>> net = resnet50(10)
        >>> param_dict = load_checkpoint("resnet50.ckpt")
        >>> load_param_into_net(net, param_dict)
        >>> gradient = Gradient(net)
    """

    def __init__(self, network):
        super(Gradient, self).__init__(network)
        self._backward_model = deepcopy(network)
        self._backward_model.set_train(False)
        self._backward_model.set_grad(False)
        self._hook_bn()
        self._grad_op = compute_gradients
        self._aggregation_fn = _abs_max


    def __call__(self, inputs, targets):
        """
        Call function for `Gradient`.

        Args:
            inputs (Tensor): The input data to be explained, a 4D tensor of shape :math:`(N, C, H, W)`.
            targets (Tensor, int): The label of interest. It should be a 1D or 0D tensor, or an integer.
                If it is a 1D tensor, its length should be the same as `inputs`.

        Returns:
            Tensor, a 4D tensor of shape :math:`(N, 1, H, W)`.

        Examples:
            >>> inputs = ms.Tensor(np.random.rand([1, 3, 224, 224]), ms.float32)
            >>> label = 5
            >>> # gradient is a Gradient object, parse data and the target label to be explained and get the attribution
            >>> saliency = gradient(inputs, label)
        """
        self._verify_data(inputs, targets)
        inputs = unify_inputs(inputs)
        targets = unify_targets(targets)

        gradient = self._grad_op(self._backward_model, *inputs, targets)
        saliency = self._aggregation_fn(gradient)
        return saliency

    def _hook_bn(self):
        """Hook BatchNorm layer for `self._backward_model.`"""
        for _, cell in self._backward_model.cells_and_names():
            if isinstance(cell, nn.BatchNorm2d):
                cache = (cell.moving_variance, cell.gamma, cell.eps)
                cell.register_backward_hook(_get_hook("2d", cache=cache))
            elif isinstance(cell, nn.BatchNorm1d):
                cache = (cell.moving_variance, cell.gamma, cell.eps)
                cell.register_backward_hook(_get_hook("1d", cache=cache))

    @staticmethod
    def _verify_data(inputs, targets):
        """Verify the validity of the parsed inputs."""
        check_value_type('inputs', inputs, Tensor)
        if len(inputs.shape) != 4:
            raise ValueError('Argument inputs must be 4D Tensor')
        check_value_type('targets', targets, (Tensor, int))
        if isinstance(targets, Tensor):
            if len(targets.shape) > 1 or (len(targets.shape) == 1 and len(targets) != len(inputs)):
                raise ValueError('Argument targets must be a 1D or 0D Tensor. If it is a 1D Tensor, '
                                 'it should have the same length as inputs.')

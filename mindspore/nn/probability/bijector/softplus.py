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
"""Softplus Bijector"""
import numpy as np
from mindspore.ops import operations as P
from mindspore.nn.layer.activation import LogSigmoid
from ..distribution._utils.custom_ops import exp_generic, expm1_generic, log_generic
from .bijector import Bijector


class Softplus(Bijector):
    r"""
    Softplus Bijector.
    This Bijector performs the operation:

    .. math::
        Y = \frac{\log(1 + e ^ {kX})}{k}
    where k is the sharpness factor.

    Args:
        sharpness (float, list, numpy.ndarray, Tensor): The scale factor. Default: 1.0.
        name (str): The name of the Bijector. Default: 'Softplus'.

    Examples:
        >>> # To initialize a Softplus bijector of sharpness 2.
        >>> softplus = nn.probability.bijector.Softplus(2)
        >>>
        >>> # To use ScalarAffine bijector in a network.
        >>> class net(Cell):
        ...     def __init__(self):
        ...         super(net, self).__init__():
        ...         self.sp1 = nn.probability.bijector.Softplus(2.)
        ...
        ...     def construct(self, value):
        ...         # Similar calls can be made to other functions
        ...         # by replacing 'forward' by the name of the function.
        ...         ans1 = self.sp1.forward(value)
        ...         ans2 = self.sp1.inverse(value)
        ...         ans3 = self.sp1.forward_log_jacobian(value)
        ...         ans4 = self.sp1.inverse_log_jacobian(value)
    """

    def __init__(self,
                 sharpness=1.0,
                 name='Softplus'):
        """
        Constructor of Softplus Bijector.
        """
        param = dict(locals())
        param['param_dict'] = {'sharpness': sharpness}
        super(Softplus, self).__init__(name=name, dtype=None, param=param)
        self._sharpness = self._add_parameter(sharpness, 'sharpness')

        self.exp = exp_generic
        self.log = log_generic
        self.expm1 = expm1_generic
        self.abs = P.Abs()
        self.dtypeop = P.DType()
        self.cast = P.Cast()
        self.fill = P.Fill()
        self.greater = P.Greater()
        self.less = P.Less()
        self.log_sigmoid = LogSigmoid()
        self.logicalor = P.LogicalOr()
        self.select = P.Select()
        self.shape = P.Shape()
        self.sigmoid = P.Sigmoid()
        self.softplus = self._softplus
        self.inverse_softplus = self._inverse_softplus

        self.threshold = np.log(np.finfo(np.float32).eps) + 1
        self.tiny = np.exp(self.threshold)

    def _softplus(self, x):
        too_small = self.less(x, self.threshold)
        too_large = self.greater(x, -self.threshold)
        too_small_value = self.exp(x)
        too_large_value = x
        ones = self.fill(self.dtypeop(x), self.shape(x), 1.0)
        too_small_or_too_large = self.logicalor(too_small, too_large)
        x = self.select(too_small_or_too_large, ones, x)
        y = self.log(self.exp(x) + 1.0)
        return self.select(too_small, too_small_value, self.select(too_large, too_large_value, y))

    def _inverse_softplus(self, x):
        r"""
        .. math::
            f(x) = \frac{\log(1 + e^{x}))}
            f^{-1}(y) = \frac{\log(e^{y} - 1)}
        """
        too_small = self.less(x, self.tiny)
        too_large = self.greater(x, -self.threshold)
        too_small_value = self.log(x)
        too_large_value = x
        ones = self.fill(self.dtypeop(x), self.shape(x), 1.0)
        too_small_or_too_large = self.logicalor(too_small, too_large)
        x = self.select(too_small_or_too_large, ones, x)
        y = x + self.log(self.abs(self.expm1(-x)))
        return self.select(too_small, too_small_value, self.select(too_large, too_large_value, y))

    @property
    def sharpness(self):
        return self._sharpness

    def extend_repr(self):
        if self.is_scalar_batch:
            str_info = f'sharpness = {self.sharpness}'
        else:
            str_info = f'batch_shape = {self.batch_shape}'
        return str_info

    def _forward(self, x):
        x = self._check_value_dtype(x)
        sharpness_local = self.cast_param_by_value(x, self.sharpness)
        scaled_value = sharpness_local * x
        forward_v = self.softplus(scaled_value) / sharpness_local
        return forward_v

    def _inverse(self, y):
        r"""
        .. math::
            f(x) = \frac{\log(1 + e^{kx}))}{k}
            f^{-1}(y) = \frac{\log(e^{ky} - 1)}{k}
        """
        y = self._check_value_dtype(y)
        sharpness_local = self.cast_param_by_value(y, self.sharpness)
        scaled_value = sharpness_local * y
        inverse_v = self.inverse_softplus(scaled_value) / sharpness_local
        return inverse_v

    def _forward_log_jacobian(self, x):
        r"""
        .. math:
            f(x) = \log(1 + e^{kx}) / k
            f'(x) = \frac{e^{kx}}{ 1 + e^{kx}}
            \log(f'(x)) =  kx - \log(1 + e^{kx}) = kx - f(kx)
        """
        x = self._check_value_dtype(x)
        sharpness_local = self.cast_param_by_value(x, self.sharpness)
        scaled_value = sharpness_local * x
        forward_log_j = self.log_sigmoid(scaled_value)
        return forward_log_j

    def _inverse_log_jacobian(self, y):
        r"""
        .. math:
            f(y) = \frac{\log(e^{ky} - 1)}{k}
            f'(y) = \frac{e^{ky}}{e^{ky} - 1}
            \log(f'(y)) = ky - \log(e^{ky} - 1) = ky - f(ky)
        """
        y = self._check_value_dtype(y)
        sharpness_local = self.cast_param_by_value(y, self.sharpness)
        scaled_value = sharpness_local * y
        inverse_log_j = scaled_value - self.inverse_softplus(scaled_value)
        return inverse_log_j

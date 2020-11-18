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
"""Scalar Affine Bijector"""
from mindspore.ops import operations as P
from ..distribution._utils.custom_ops import log_generic
from .bijector import Bijector


class ScalarAffine(Bijector):
    """
    Scalar Affine Bijector.
    This Bijector performs the operation:

    .. math::
        Y = a * X + b
    where a is the scale factor and b is the shift factor.

    Args:
        scale (float, list, numpy.ndarray, Tensor): The scale factor. Default: 1.0.
        shift (float, list, numpy.ndarray, Tensor): The shift factor. Default: 0.0.
        name (str): The name of the bijector. Default: 'ScalarAffine'.

    Note:
        If `shift`, `scale` are passed in as numpy.ndarray or tensor, they have to have
        the same dtype otherwise an error will be raised.

    Examples:
        >>> # To initialize a ScalarAffine bijector of scale 1 and shift 2.
        >>> scalaraffine = nn.probability.bijector.ScalarAffine(1, 2)
        >>>
        >>> # To use a ScalarAffine bijector in a network.
        >>> class net(Cell):
        ...     def __init__(self):
        ...         super(net, self).__init__():
        ...         self.s1 = nn.probability.bijector.ScalarAffine(1, 2)
        ...
        ...     def construct(self, value):
        ...         # Similar calls can be made to other functions
        ...         # by replacing 'forward' by the name of the function.
        ...         ans1 = self.s1.forward(value)
        ...         ans2 = self.s1.inverse(value)
        ...         ans3 = self.s1.forward_log_jacobian(value)
        ...         ans4 = self.s1.inverse_log_jacobian(value)
    """

    def __init__(self,
                 scale=1.0,
                 shift=0.0,
                 name='ScalarAffine'):
        """
        Constructor of ScalarAffine Bijector.
        """
        param = dict(locals())
        param['param_dict'] = {'scale': scale, 'shift': shift}
        super(ScalarAffine, self).__init__(
            is_constant_jacobian=True,
            is_injective=True,
            name=name,
            dtype=None,
            param=param)

        self._scale = self._add_parameter(scale, 'scale')
        self._shift = self._add_parameter(shift, 'shift')

        self.abs = P.Abs()
        self.oneslike = P.OnesLike()
        self.dtypeop = P.DType()
        self.cast = P.Cast()
        self.log = log_generic

    @property
    def scale(self):
        return self._scale

    @property
    def shift(self):
        return self._shift

    def extend_repr(self):
        if self.is_scalar_batch:
            str_info = f'scale = {self.scale}, shift = {self.shift}'
        else:
            str_info = f'batch_shape = {self.batch_shape}'
        return str_info

    def _forward(self, x):
        r"""
        .. math::
            f(x) = a * x + b
        """
        x = self._check_value_dtype(x)
        scale_local = self.cast_param_by_value(x, self.scale)
        shift_local = self.cast_param_by_value(x, self.shift)
        forward_v = scale_local * x + shift_local * self.oneslike(x)
        return forward_v

    def _inverse(self, y):
        r"""
        .. math::
            f(y) = \frac{y - b}{a}
        """
        y = self._check_value_dtype(y)
        scale_local = self.cast_param_by_value(y, self.scale)
        shift_local = self.cast_param_by_value(y, self.shift)
        inverse_v = (y - shift_local) / scale_local
        return inverse_v

    def _forward_log_jacobian(self, x):
        r"""
        .. math::
            f(x) = a * x + b
            f'(x) = a
            \log(f'(x)) = \log(a)
        """
        x = self._check_value_dtype(x)
        scale_local = self.cast_param_by_value(x, self.scale)
        forward_log_j = self.log(self.abs(scale_local))
        return forward_log_j

    def _inverse_log_jacobian(self, y):
        r"""
        .. math::
            f(y) = \frac{(y - b)}{a}
            f'(x) = \frac{1.0}{a}
            \log(f'(x)) = - \log(a)
        """
        y = self._check_value_dtype(y)
        scale_local = self.cast_param_by_value(y, self.scale)
        inverse_log_j = -1. * self.log(self.abs(scale_local))
        return inverse_log_j

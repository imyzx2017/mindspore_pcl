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
"""Cauchy Distribution"""
import numpy as np
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore._checkparam import Validator
from mindspore.common import dtype as mstype
from .distribution import Distribution
from ._utils.utils import check_greater_zero, check_distribution_name, raise_not_defined
from ._utils.custom_ops import exp_generic, log_generic, log1p_generic


class Cauchy(Distribution):
    """
    Cauchy distribution.

    Args:
        loc (int, float, list, numpy.ndarray, Tensor, Parameter): The location of the Cauchy distribution.
        scale (int, float, list, numpy.ndarray, Tensor, Parameter): The scale of the Cauchy distribution.
        seed (int): The seed used in sampling. The global seed is used if it is None. Default: None.
        dtype (mindspore.dtype): The type of the event samples. Default: mstype.float32.
        name (str): The name of the distribution. Default: 'Cauchy'.

    Note:
        `scale` must be greater than zero.
        `dist_spec_args` are `loc` and `scale`.
        `dtype` must be a float type because Cauchy distributions are continuous.
        Cauchy distribution is not supported on GPU backend.

    Examples:
        >>> # To initialize a Cauchy distribution of loc 3.0 and scale 4.0.
        >>> import mindspore.nn.probability.distribution as msd
        >>> cauchy = msd.Cauchy(3.0, 4.0, dtype=mstype.float32)
        >>>
        >>> # The following creates two independent Cauchy distributions.
        >>> cauchy = msd.Cauchy([3.0, 3.0], [4.0, 4.0], dtype=mstype.float32)
        >>>
        >>> # A Cauchy distribution can be initilize without arguments.
        >>> # In this case, 'loc' and `scale` must be passed in through arguments.
        >>> cauchy = msd.Cauchy(dtype=mstype.float32)
        >>>
        >>> # To use a Cauchy distribution in a network.
        >>> class net(Cell):
        >>>     def __init__(self):
        >>>         super(net, self).__init__():
        >>>         self.cau1 = msd.Cauchy(0.0, 1.0, dtype=mstype.float32)
        >>>         self.cau2 = msd.Cauchy(dtype=mstype.float32)
        >>>
        >>>     # The following calls are valid in construct.
        >>>     def construct(self, value, loc_b, scale_b, loc_a, scale_a):
        >>>
        >>>         # Private interfaces of probability functions corresponding to public interfaces, including
        >>>         # `prob`, `log_prob`, `cdf`, `log_cdf`, `survival_function`, and `log_survival`, have the same arguments as follows.
        >>>         # Args:
        >>>         #     value (Tensor): the value to be evaluated.
        >>>         #     loc (Tensor): the location of the distribution. Default: self.loc.
        >>>         #     scale (Tensor): the scale of the distribution. Default: self.scale.
        >>>
        >>>         # Examples of `prob`.
        >>>         # Similar calls can be made to other probability functions
        >>>         # by replacing 'prob' by the name of the function
        >>>         ans = self.cau1.prob(value)
        >>>         # Evaluate with respect to distribution b.
        >>>         ans = self.cau1.prob(value, loc_b, scale_b)
        >>>         # `loc` and `scale` must be passed in during function calls
        >>>         ans = self.cau2.prob(value, loc_a, scale_a)
        >>>
        >>>         # Functions `mode` and `entropy` have the same arguments.
        >>>         # Args:
        >>>         #     loc (Tensor): the location of the distribution. Default: self.loc.
        >>>         #     scale (Tensor): the scale of the distribution. Default: self.scale.
        >>>
        >>>         # Example of `mode`.
        >>>         ans = self.cau1.mode() # return 0.0
        >>>         ans = self.cau1.mode(loc_b, scale_b) # return loc_b
        >>>         # `loc` and `scale` must be passed in during function calls.
        >>>         ans = self.cau2.mode(loc_a, scale_a)
        >>>
        >>>         # Interfaces of 'kl_loss' and 'cross_entropy' are the same:
        >>>         # Args:
        >>>         #     dist (str): the type of the distributions. Only "Cauchy" is supported.
        >>>         #     loc_b (Tensor): the loc of distribution b.
        >>>         #     scale_b (Tensor): the scale distribution b.
        >>>         #     loc (Tensor): the loc of distribution a. Default: self.loc.
        >>>         #     scale (Tensor): the scale distribution a. Default: self.scale.
        >>>
        >>>         # Examples of `kl_loss`. `cross_entropy` is similar.
        >>>         ans = self.cau1.kl_loss('Cauchy', loc_b, scale_b)
        >>>         ans = self.cau1.kl_loss('Cauchy', loc_b, scale_b, loc_a, scale_a)
        >>>         # Additional `loc` and `scale` must be passed in.
        >>>         ans = self.cau2.kl_loss('Cauchy', loc_b, scale_b, loc_a, scale_a)
        >>>
        >>>         # Examples of `sample`.
        >>>         # Args:
        >>>         #     shape (tuple): the shape of the sample. Default: ()
        >>>         #     loc (Tensor): the location of the distribution. Default: self.loc.
        >>>         #     scale (Tensor): the scale of the distribution. Default: self.scale.
        >>>         ans = self.cau1.sample()
        >>>         ans = self.cau1.sample((2,3))
        >>>         ans = self.cau1.sample((2,3), loc_b, s_b)
        >>>         ans = self.cau2.sample((2,3), loc_a, s_a)
    """

    def __init__(self,
                 loc=None,
                 scale=None,
                 seed=None,
                 dtype=mstype.float32,
                 name="Cauchy"):
        """
        Constructor of Cauchy.
        """
        param = dict(locals())
        param['param_dict'] = {'loc': loc, 'scale': scale}
        valid_dtype = mstype.float_type
        Validator.check_type_name("dtype", dtype, valid_dtype, type(self).__name__)
        super(Cauchy, self).__init__(seed, dtype, name, param)

        self._loc = self._add_parameter(loc, 'loc')
        self._scale = self._add_parameter(scale, 'scale')
        if self._scale is not None:
            check_greater_zero(self._scale, "scale")

        # ops needed for the class
        self.atan = P.Atan()
        self.cast = P.Cast()
        self.const = P.ScalarToArray()
        self.dtypeop = P.DType()
        self.exp = exp_generic
        self.fill = P.Fill()
        self.less = P.Less()
        self.log = log_generic
        self.log1p = log1p_generic
        self.squeeze = P.Squeeze(0)
        self.shape = P.Shape()
        self.sq = P.Square()
        self.sqrt = P.Sqrt()
        self.tan = P.Tan()
        self.uniform = C.uniform


    def extend_repr(self):
        if self.is_scalar_batch:
            str_info = f'location = {self._loc}, scale = {self._scale}'
        else:
            str_info = f'batch_shape = {self._broadcast_shape}'
        return str_info

    @property
    def loc(self):
        """
        Return the location of the distribution.
        """
        return self._loc

    @property
    def scale(self):
        """
        Return the scale of the distribution.
        """
        return self._scale

    def _get_dist_type(self):
        return "Cauchy"

    def _get_dist_args(self, loc=None, scale=None):
        if loc is not None:
            self.checktensor(loc, 'loc')
        else:
            loc = self.loc
        if scale is not None:
            self.checktensor(scale, 'scale')
        else:
            scale = self.scale
        return loc, scale

    def _mode(self, loc=None, scale=None):
        """
        The mode of the distribution.
        """
        loc, scale = self._check_param_type(loc, scale)
        return loc

    def _mean(self, *args, **kwargs):
        return raise_not_defined('mean', 'Cauchy', *args, **kwargs)

    def _sd(self, *args, **kwargs):
        return raise_not_defined('standard deviation', 'Cauchy', *args, **kwargs)

    def _var(self, *args, **kwargs):
        return raise_not_defined('variance', 'Cauchy', *args, **kwargs)

    def _entropy(self, loc=None, scale=None):
        r"""
        Evaluate entropy.

        .. math::
            H(X) = \log(4 * \Pi * scale)
        """
        loc, scale = self._check_param_type(loc, scale)
        return self.log(4 * np.pi * scale)

    def _log_prob(self, value, loc=None, scale=None):
        r"""
        Evaluate log probability.

        Args:
            value (Tensor): The value to be evaluated.
            loc (Tensor): The location of the distribution. Default: self.loc.
            scale (Tensor): The scale of the distribution. Default: self.scale.

        .. math::
            L(x) = \log(\frac{1}{\pi * scale} * \frac{scale^{2}}{(x - loc)^{2} + scale^{2}})
        """
        value = self._check_value(value, 'value')
        value = self.cast(value, self.dtype)
        loc, scale = self._check_param_type(loc, scale)
        z = (value - loc) / scale
        log_unnormalized_prob = - self.log1p(self.sq(z))
        log_normalization = self.log(np.pi * scale)
        return log_unnormalized_prob - log_normalization

    def _cdf(self, value, loc=None, scale=None):
        r"""
        Evaluate the cumulative distribution function on the given value.

        Args:
            value (Tensor): The value to be evaluated.
            loc (Tensor): The location of the distribution. Default: self.loc.
            scale (Tensor): The scale the distribution. Default: self.scale.

        .. math::
            cdf(x) = \frac{\arctan{(x - loc) / scale}}{\pi} + 0.5
        """
        value = self._check_value(value, 'value')
        value = self.cast(value, self.dtype)
        loc, scale = self._check_param_type(loc, scale)
        z = (value - loc) / scale
        return self.atan(z) / np.pi + 0.5

    def _log_cdf(self, value, loc=None, scale=None):
        r"""
        Evaluate the log cumulative distribution function on the given value.

        Args:
            value (Tensor): The value to be evaluated.
            loc (Tensor): The location of the distribution. Default: self.loc.
            scale (Tensor): The scale the distribution. Default: self.scale.

        .. math::
            log_cdf(x) = \log(\frac{\arctan(\frac{x-loc}{scale})}{\pi} + 0.5)
                       = \log {\arctan(\frac{x-loc}{scale}) + 0.5pi}{pi}
                       = \log1p \frac{2 * arctan(\frac{x-loc}{scale})}{pi} - \log2
        """
        value = self._check_value(value, 'value')
        value = self.cast(value, self.dtype)
        loc, scale = self._check_param_type(loc, scale)
        z = (value - loc) / scale
        return self.log1p(2. * self.atan(z) / np.pi) - self.log(self.const(2.))

    def _quantile(self, p, loc=None, scale=None):
        loc, scale = self._check_param_type(loc, scale)
        return loc + scale * self.tan(np.pi * (p - 0.5))

    def _kl_loss(self, dist, loc_b, scale_b, loc=None, scale=None):
        r"""
        Evaluate Cauchy-Cauchy kl divergence, i.e. KL(a||b).

        Args:
            dist (str): The type of the distributions. Should be "Cauchy" in this case.
            loc_b (Tensor): The loc of distribution b.
            scale_b (Tensor): The scale of distribution b.
            loc (Tensor): The loc of distribution a. Default: self.loc.
            scale (Tensor): The scale of distribution a. Default: self.scale.

        .. math::
            KL(a||b) = \log(\frac{(scale_a + scale_b)^{2} + (loc_a - loc_b)^{2}}
                                 {4 * scale_a * scale_b})
        """
        check_distribution_name(dist, 'Cauchy')
        loc, scale = self._check_param_type(loc, scale)
        loc_b = self._check_value(loc_b, 'loc_b')
        loc_b = self.cast(loc_b, self.parameter_type)
        scale_b = self._check_value(scale_b, 'scale_b')
        scale_b = self.cast(scale_b, self.parameter_type)
        sum_square = self.sq(scale + scale_b)
        square_diff = self.sq(loc - loc_b)
        return self.log(sum_square + square_diff) - \
                self.log(self.const(4.0)) - self.log(scale) - self.log(scale_b)

    def _cross_entropy(self, dist, loc_b, scale_b, loc=None, scale=None):
        r"""
        Evaluate cross entropy between Cauchy distributions.

        Args:
            dist (str): The type of the distributions. Should be "Cauchy" in this case.
            loc_b (Tensor): The loc of distribution b.
            scale_b (Tensor): The scale of distribution b.
            loc (Tensor): The loc of distribution a. Default: self.loc.
            scale (Tensor): The scale of distribution a. Default: self.scale.
        """
        check_distribution_name(dist, 'Cauchy')
        return self._entropy(loc, scale) + self._kl_loss(dist, loc_b, scale_b, loc, scale)

    def _sample(self, shape=(), loc=None, scale=None):
        """
        Sampling.

        Args:
            shape (tuple): The shape of the sample. Default: ().
            loc (Tensor): The location of the samples. Default: self.loc.
            scale (Tensor): The scale of the samples. Default: self.scale.

        Returns:
            Tensor, with the shape being shape + batch_shape.
        """
        shape = self.checktuple(shape, 'shape')
        loc, scale = self._check_param_type(loc, scale)
        batch_shape = self.shape(loc + scale)
        origin_shape = shape + batch_shape
        if origin_shape == ():
            sample_shape = (1,)
        else:
            sample_shape = origin_shape
        l_zero = self.const(0.0)
        h_one = self.const(1.0)
        sample_uniform = self.uniform(sample_shape, l_zero, h_one, self.seed)
        sample = self._quantile(sample_uniform, loc, scale)
        value = self.cast(sample, self.dtype)
        if origin_shape == ():
            value = self.squeeze(value)
        return value

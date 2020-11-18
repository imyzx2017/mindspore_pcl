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
"""Normal Distribution"""
import numpy as np
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore._checkparam import Validator
from mindspore.common import dtype as mstype
from .distribution import Distribution
from ._utils.utils import check_greater_zero, check_distribution_name
from ._utils.custom_ops import exp_generic, expm1_generic, log_generic


class Normal(Distribution):
    """
    Normal distribution.

    Args:
        mean (int, float, list, numpy.ndarray, Tensor, Parameter): The mean of the Normal distribution.
        sd (int, float, list, numpy.ndarray, Tensor, Parameter): The standard deviation of the Normal distribution.
        seed (int): The seed used in sampling. The global seed is used if it is None. Default: None.
        dtype (mindspore.dtype): The type of the event samples. Default: mstype.float32.
        name (str): The name of the distribution. Default: 'Normal'.

    Note:
        `sd` must be greater than zero.
        `dist_spec_args` are `mean` and `sd`.
        `dtype` must be a float type because Normal distributions are continuous.

    Examples:
        >>> # To initialize a Normal distribution of the mean 3.0 and the standard deviation 4.0.
        >>> import mindspore.nn.probability.distribution as msd
        >>> n = msd.Normal(3.0, 4.0, dtype=mstype.float32)
        >>>
        >>> # The following creates two independent Normal distributions.
        >>> n = msd.Normal([3.0, 3.0], [4.0, 4.0], dtype=mstype.float32)
        >>>
        >>> # A Normal distribution can be initilize without arguments.
        >>> # In this case, `mean` and `sd` must be passed in through arguments.
        >>> n = msd.Normal(dtype=mstype.float32)
        >>>
        >>> # To use a Normal distribution in a network.
        >>> class net(Cell):
        >>>     def __init__(self):
        >>>         super(net, self).__init__():
        >>>         self.n1 = msd.Nomral(0.0, 1.0, dtype=mstype.float32)
        >>>         self.n2 = msd.Normal(dtype=mstype.float32)
        >>>
        >>>     # The following calls are valid in construct.
        >>>     def construct(self, value, mean_b, sd_b, mean_a, sd_a):
        >>>
        >>>         # Private interfaces of probability functions corresponding to public interfaces, including
        >>>         # `prob`, `log_prob`, `cdf`, `log_cdf`, `survival_function`, and `log_survival`, have the same arguments as follows.
        >>>         # Args:
        >>>         #     value (Tensor): the value to be evaluated.
        >>>         #     mean (Tensor): the mean of distribution. Default: self._mean_value.
        >>>         #     sd (Tensor): the standard deviation of distribution. Default: self._sd_value.
        >>>
        >>>         # Examples of `prob`.
        >>>         # Similar calls can be made to other probability functions
        >>>         # by replacing 'prob' by the name of the function
        >>>         ans = self.n1.prob(value)
        >>>         # Evaluate with respect to distribution b.
        >>>         ans = self.n1.prob(value, mean_b, sd_b)
        >>>         # `mean` and `sd` must be passed in during function calls
        >>>         ans = self.n2.prob(value, mean_a, sd_a)
        >>>
        >>>
        >>>         # Functions `mean`, `sd`, `var`, and `entropy` have the same arguments.
        >>>         # Args:
        >>>         #     mean (Tensor): the mean of distribution. Default: self._mean_value.
        >>>         #     sd (Tensor): the standard deviation of distribution. Default: self._sd_value.
        >>>
        >>>         # Example of `mean`. `sd`, `var`, and `entropy` are similar.
        >>>         ans = self.n1.mean() # return 0.0
        >>>         ans = self.n1.mean(mean_b, sd_b) # return mean_b
        >>>         # `mean` and `sd` must be passed in during function calls.
        >>>         ans = self.n2.mean(mean_a, sd_a)
        >>>
        >>>
        >>>         # Interfaces of 'kl_loss' and 'cross_entropy' are the same:
        >>>         # Args:
        >>>         #     dist (str): the type of the distributions. Only "Normal" is supported.
        >>>         #     mean_b (Tensor): the mean of distribution b.
        >>>         #     sd_b (Tensor): the standard deviation distribution b.
        >>>         #     mean_a (Tensor): the mean of distribution a. Default: self._mean_value.
        >>>         #     sd_a (Tensor): the standard deviation distribution a. Default: self._sd_value.
        >>>
        >>>         # Examples of `kl_loss`. `cross_entropy` is similar.
        >>>         ans = self.n1.kl_loss('Normal', mean_b, sd_b)
        >>>         ans = self.n1.kl_loss('Normal', mean_b, sd_b, mean_a, sd_a)
        >>>         # Additional `mean` and `sd` must be passed in.
        >>>         ans = self.n2.kl_loss('Normal', mean_b, sd_b, mean_a, sd_a)
        >>>
        >>>         # Examples of `sample`.
        >>>         # Args:
        >>>         #     shape (tuple): the shape of the sample. Default: ()
        >>>         #     mean (Tensor): the mean of the distribution. Default: self._mean_value.
        >>>         #     sd (Tensor): the standard deviation of the distribution. Default: self._sd_value.
        >>>         ans = self.n1.sample()
        >>>         ans = self.n1.sample((2,3))
        >>>         ans = self.n1.sample((2,3), mean_b, sd_b)
        >>>         ans = self.n2.sample((2,3), mean_a, sd_a)
    """

    def __init__(self,
                 mean=None,
                 sd=None,
                 seed=None,
                 dtype=mstype.float32,
                 name="Normal"):
        """
        Constructor of Normal.
        """
        param = dict(locals())
        param['param_dict'] = {'mean': mean, 'sd': sd}
        valid_dtype = mstype.float_type
        Validator.check_type_name("dtype", dtype, valid_dtype, type(self).__name__)
        super(Normal, self).__init__(seed, dtype, name, param)

        self._mean_value = self._add_parameter(mean, 'mean')
        self._sd_value = self._add_parameter(sd, 'sd')
        if self._sd_value is not None:
            check_greater_zero(self._sd_value, "Standard deviation")

        # ops needed for the class
        self.exp = exp_generic
        self.expm1 = expm1_generic
        self.log = log_generic
        self.erf = P.Erf()
        self.squeeze = P.Squeeze(0)
        self.cast = P.Cast()
        self.const = P.ScalarToArray()
        self.shape = P.Shape()
        self.sq = P.Square()
        self.sqrt = P.Sqrt()

    def extend_repr(self):
        if self.is_scalar_batch:
            s = f'mean = {self._mean_value}, standard deviation = {self._sd_value}'
        else:
            s = f'batch_shape = {self._broadcast_shape}'
        return s

    def _get_dist_type(self):
        return "Normal"

    def _get_dist_args(self, mean=None, sd=None):
        if mean is not None:
            self.checktensor(mean, 'mean')
        else:
            mean = self._mean_value
        if sd is not None:
            self.checktensor(sd, 'sd')
        else:
            sd = self._sd_value
        return mean, sd

    def _mean(self, mean=None, sd=None):
        """
        The mean of the distribution.
        """
        mean, sd = self._check_param_type(mean, sd)
        return mean

    def _mode(self, mean=None, sd=None):
        """
        The mode of the distribution.
        """
        mean, sd = self._check_param_type(mean, sd)
        return mean

    def _sd(self, mean=None, sd=None):
        """
        The standard deviation of the distribution.
        """
        mean, sd = self._check_param_type(mean, sd)
        return sd

    def _entropy(self, mean=None, sd=None):
        r"""
        Evaluate entropy.

        .. math::
            H(X) = \log(\sqrt(numpy.e * 2. * numpy.pi * \sq(\sigma)))
        """
        mean, sd = self._check_param_type(mean, sd)
        return self.log(self.sqrt(self.const(np.e * 2. * np.pi))) + self.log(sd)

    def _cross_entropy(self, dist, mean_b, sd_b, mean=None, sd=None):
        r"""
        Evaluate cross entropy between normal distributions.

        Args:
            dist (str): Type of the distributions. Should be "Normal" in this case.
            mean_b (Tensor): Mean of distribution b.
            sd_b (Tensor): Standard deviation distribution b.
            mean_a (Tensor): Mean of distribution a. Default: self._mean_value.
            sd_a (Tensor): Standard deviation distribution a. Default: self._sd_value.
        """
        check_distribution_name(dist, 'Normal')
        return self._entropy(mean, sd) + self._kl_loss(dist, mean_b, sd_b, mean, sd)

    def _log_prob(self, value, mean=None, sd=None):
        r"""
        Evaluate log probability.

        Args:
            value (Tensor): The value to be evaluated.
            mean (Tensor): The mean of the distribution. Default: self._mean_value.
            sd (Tensor): The standard deviation the distribution. Default: self._sd_value.

        .. math::
            L(x) = -1* \frac{(x - \mu)^2}{2. * \sigma^2} - \log(\sqrt(2* \pi * \sigma^2))
        """
        value = self._check_value(value, 'value')
        value = self.cast(value, self.dtype)
        mean, sd = self._check_param_type(mean, sd)
        unnormalized_log_prob = -1. * \
            (self.sq(value - mean)) / (2. * self.sq(sd))
        neg_normalization = -1. * \
            self.log(self.const(2. * np.pi)) / 2. - self.log(sd)
        return unnormalized_log_prob + neg_normalization

    def _cdf(self, value, mean=None, sd=None):
        r"""
        Evaluate the cumulative distribution function on the given value.

        Args:
            value (Tensor): The value to be evaluated.
            mean (Tensor): The mean of the distribution. Default: self._mean_value.
            sd (Tensor): The standard deviation the distribution. Default: self._sd_value.

        .. math::
            cdf(x) = 0.5 * (1+ Erf((x - \mu) / ( \sigma * \sqrt(2))))
        """
        value = self._check_value(value, 'value')
        value = self.cast(value, self.dtype)
        mean, sd = self._check_param_type(mean, sd)
        sqrt2 = self.sqrt(self.const(2.0))
        adjusted = (value - mean) / (sd * sqrt2)
        return 0.5 * (1.0 + self.erf(adjusted))

    def _kl_loss(self, dist, mean_b, sd_b, mean=None, sd=None):
        r"""
        Evaluate Normal-Normal KL divergence, i.e. KL(a||b).

        Args:
            dist (str): The type of the distributions. Should be "Normal" in this case.
            mean_b (Tensor): The mean of distribution b.
            sd_b (Tensor): The standard deviation distribution b.
            mean_a (Tensor): The mean of distribution a. Default: self._mean_value.
            sd_a (Tensor): The standard deviation distribution a. Default: self._sd_value.

        .. math::
            KL(a||b) = 0.5 * (\frac{MEAN(a)}{STD(b)} - \frac{MEAN(b)}{STD(b)}) ^ 2 +
                       0.5 * EXPM1(2 * (\log(STD(a)) - \log(STD(b))) - (\log(STD(a)) - \log(STD(b)))
        """
        check_distribution_name(dist, 'Normal')
        mean_b = self._check_value(mean_b, 'mean_b')
        sd_b = self._check_value(sd_b, 'sd_b')
        mean_b = self.cast(mean_b, self.parameter_type)
        sd_b = self.cast(sd_b, self.parameter_type)
        mean_a, sd_a = self._check_param_type(mean, sd)
        diff_log_scale = self.log(sd_a) - self.log(sd_b)
        squared_diff = self.sq(mean_a / sd_b - mean_b / sd_b)
        return 0.5 * squared_diff + 0.5 * self.expm1(2 * diff_log_scale) - diff_log_scale

    def _sample(self, shape=(), mean=None, sd=None):
        """
        Sampling.

        Args:
            shape (tuple): The shape of the sample. Default: ().
            mean (Tensor): The mean of the samples. Default: self._mean_value.
            sd (Tensor): The standard deviation of the samples. Default: self._sd_value.

        Returns:
            Tensor, with the shape being shape + batch_shape.
        """
        shape = self.checktuple(shape, 'shape')
        mean, sd = self._check_param_type(mean, sd)
        batch_shape = self.shape(mean + sd)
        origin_shape = shape + batch_shape
        if origin_shape == ():
            sample_shape = (1,)
        else:
            sample_shape = origin_shape
        sample_norm = C.normal(sample_shape, mean, sd, self.seed)
        value = self.cast(sample_norm, self.dtype)
        if origin_shape == ():
            value = self.squeeze(value)
        return value

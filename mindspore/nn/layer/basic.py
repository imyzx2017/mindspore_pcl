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

"""basic"""

import numpy as np
import mindspore.common.dtype as mstype
from mindspore.common.seed import _get_graph_seed
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.functional import identity
from mindspore.ops.operations import _inner_ops as inner
from mindspore.ops.primitive import constexpr, Primitive
from mindspore.common.parameter import Parameter
from mindspore._extends import cell_attr_register
from mindspore._checkparam import Rel, Validator
from mindspore.common.api import ms_function
from mindspore import context
from ..cell import Cell
from .activation import get_activation


__all__ = ['Dropout', 'Flatten', 'Dense', 'ClipByNorm', 'Norm', 'OneHot', 'Pad', 'Unfold',
           'MatrixDiag', 'MatrixDiagPart', 'MatrixSetDiag']


class Dropout(Cell):
    r"""
    Dropout layer for the input.

    Randomly set some elements of the input tensor to zero with probability :math:`1 - keep\_prob` during training
    using samples from a Bernoulli distribution.

    Note:
        Each channel will be zeroed out independently on every construct call.

        The outputs are scaled by a factor of :math:`\frac{1}{keep\_prob}` during training so
        that the output layer remains at a similar scale. During inference, this
        layer returns the same tensor as the input.

        This technique is proposed in paper `Dropout: A Simple Way to Prevent Neural Networks from Overfitting
        <http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf>`_ and proved to be effective to reduce
        over-fitting and prevents neurons from co-adaptation. See more details in `Improving neural networks by
        preventing co-adaptation of feature detectors
        <https://arxiv.org/pdf/1207.0580.pdf>`_.

    Args:
        keep_prob (float): The keep rate, greater than 0 and less equal than 1. E.g. rate=0.9,
                   dropping out 10% of input units. Default: 0.5.
        dtype (:class:`mindspore.dtype`): Data type of input. Default: mindspore.float32.

    Raises:
        ValueError: If `keep_prob` is not in range (0, 1].

    Inputs:
        - **input** (Tensor) - The input tensor.

    Outputs:
        Tensor, output tensor with the same shape as the input.

    Examples:
        >>> x = Tensor(np.ones([2, 2, 3]), mindspore.float32)
        >>> net = nn.Dropout(keep_prob=0.8)
        >>> net.set_train()
        >>> net(x)
        [[[0., 1.25, 0.],
          [1.25, 1.25, 1.25]],
         [[1.25, 1.25, 1.25],
          [1.25, 1.25, 1.25]]]
    """

    def __init__(self, keep_prob=0.5, dtype=mstype.float32):
        super(Dropout, self).__init__()
        if keep_prob <= 0 or keep_prob > 1:
            raise ValueError("dropout probability should be a number in range (0, 1], but got {}".format(keep_prob))
        Validator.check_subclass("dtype", dtype, mstype.number_type, self.cls_name)
        Validator.check_value_type('keep_prob', keep_prob, [float], self.cls_name)
        self.keep_prob = keep_prob
        seed0, seed1 = _get_graph_seed(0, "dropout")
        self.seed0 = seed0
        self.seed1 = seed1
        self.dtype = dtype
        self.get_shape = P.Shape()
        self.dropout_gen_mask = P.DropoutGenMask(Seed0=self.seed0, Seed1=self.seed1)
        self.dropout_do_mask = P.DropoutDoMask()
        self.cast = P.Cast()
        self.is_gpu = context.get_context('device_target') in ["GPU"]
        self.dropout = P.Dropout(keep_prob)

    def construct(self, x):
        if not self.training:
            return x

        if self.is_gpu:
            out, _ = self.dropout(x)
            return out

        if self.keep_prob == 1:
            return x

        shape = self.get_shape(x)
        dtype = P.DType()(x)
        if _is_float_dtype(dtype):
            keep_prob = self.cast(self.keep_prob, dtype)
        else:
            keep_prob = self.cast(self.keep_prob, mstype.float16)
        output = self.dropout_gen_mask(shape, keep_prob)
        return self.dropout_do_mask(x, output, keep_prob)

    def extend_repr(self):
        return 'keep_prob={}, dtype={}'.format(self.keep_prob, self.dtype)


class Flatten(Cell):
    r"""
    Flatten layer for the input.

    Flattens a tensor without changing dimension of batch size on the 0-th axis.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, \ldots)` to be flattened.

    Outputs:
        Tensor, the shape of the output tensor is :math:`(N, X)`, where :math:`X` is
        the product of the remaining dimensions.

    Examples:
        >>> input = Tensor(np.array([[[1.2, 1.2], [2.1, 2.1]], [[2.2, 2.2], [3.2, 3.2]]]), mindspore.float32)
        >>> net = nn.Flatten()
        >>> net(input)
        [[1.2 1.2 2.1 2.1]
         [2.2 2.2 3.2 3.2]]
    """

    def __init__(self):
        super(Flatten, self).__init__()

    def construct(self, x):
        return F.reshape(x, (F.shape(x)[0], -1))

@constexpr
def get_broadcast_weight_bias_shape(x_shape, out_channel, in_channel):
    """get broadcast_weight_bias shape"""
    broad_weight_shape = x_shape[:-2] + (out_channel, in_channel)
    broad_bias_shape = x_shape[:-1] + (out_channel,)
    return broad_weight_shape, broad_bias_shape

class Dense(Cell):
    r"""
    The dense connected layer.

    Applies dense connected layer for the input. This layer implements the operation as:

    .. math::
        \text{outputs} = \text{activation}(\text{inputs} * \text{kernel} + \text{bias}),

    where :math:`\text{activation}` is the activation function passed as the activation
    argument (if passed in), :math:`\text{kernel}` is a weight matrix with the same
    data type as the inputs created by the layer, and :math:`\text{bias}` is a bias vector
    with the same data type as the inputs created by the layer (only if has_bias is True).

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as input x. The values of str refer to the function `initializer`. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The dtype is
            same as input x. The values of str refer to the function `initializer`. Default: 'zeros'.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: True.
        activation (Union[str, Cell, Primitive]): activate function applied to the output of the fully connected layer,
            eg. 'ReLU'.Default: None.

    Raises:
        ValueError: If weight_init or bias_init shape is incorrect.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, in\_channels)`.

    Outputs:
        Tensor of shape :math:`(N, out\_channels)`.

    Examples:
        >>> input = Tensor(np.random.randint(0, 255, [2, 3]), mindspore.float32)
        >>> net = nn.Dense(3, 4)
        >>> net(input)
        [[ 2.5246444   2.2738023   0.5711005  -3.9399147 ]
         [ 1.0739875   4.0155234   0.94188046 -5.459526  ]]
    """
    @cell_attr_register(attrs=['has_bias', 'activation'])
    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias=True,
                 activation=None):
        super(Dense, self).__init__()
        self.in_channels = Validator.check_positive_int(in_channels)
        self.out_channels = Validator.check_positive_int(out_channels)
        self.has_bias = Validator.check_bool(has_bias)
        self.shape_op = P.Shape()

        if isinstance(weight_init, Tensor):
            if weight_init.dim() != 2 or weight_init.shape[0] != out_channels or \
               weight_init.shape[1] != in_channels:
                raise ValueError("Weight init shape error.")
        self.weight = Parameter(initializer(weight_init, [out_channels, in_channels]), name="weight")

        self.bias = None
        if self.has_bias:
            if isinstance(bias_init, Tensor):
                if bias_init.dim() != 1 or bias_init.shape[0] != out_channels:
                    raise ValueError("Bias init shape error.")
            self.bias = Parameter(initializer(bias_init, [out_channels]), name="bias")
            self.bias_add = P.BiasAdd()
            self.tensor_add = P.TensorAdd()

        self.matmul = P.MatMul(transpose_b=True)
        self.batch_matmul = P.BatchMatMul(transpose_b=True)
        self.activation = get_activation(activation) if isinstance(activation, str) else activation
        if activation is not None and not isinstance(self.activation, (Cell, Primitive)):
            raise TypeError("The activation must be str or Cell or Primitive,"" but got {}.".format(activation))
        self.activation_flag = self.activation is not None

    def construct(self, x):
        x_shape = self.shape_op(x)
        x_dim = len(x_shape)
        if x_dim == 2:
            matmul = self.matmul
            bias_add = self.bias_add if self.has_bias else None
            weight = self.weight
            bias = self.bias
        else:
            broad_weight_shape, broad_bias_shape = get_broadcast_weight_bias_shape(x_shape, self.out_channels,
                                                                                   self.in_channels)
            weight_broadcast_to = P.BroadcastTo(broad_weight_shape)
            bias_broadcast_to = P.BroadcastTo(broad_bias_shape)
            matmul = self.batch_matmul
            bias_add = self.tensor_add if self.has_bias else None
            weight = weight_broadcast_to(self.weight)
            bias = bias_broadcast_to(self.bias) if self.has_bias else self.bias

        x = matmul(x, weight)
        if self.has_bias:
            x = bias_add(x, bias)
        if self.activation_flag:
            x = self.activation(x)
        return x


    def extend_repr(self):
        s = 'input_channels={}, output_channels={}'.format(self.in_channels, self.out_channels)
        if self.has_bias:
            s += ', has_bias={}'.format(self.has_bias)
        if self.activation_flag:
            s += ', activation={}'.format(self.activation)
        return s


@constexpr
def _is_equal_one(x):
    if x is None:
        return False
    return bool(x.asnumpy().mean() == 1.0)

@constexpr
def _dtype_check(x_dtype):
    if x_dtype not in [mstype.float32, mstype.float16]:
        raise  TypeError("The input type must be float32 or float16.")

@constexpr
def _is_float_dtype(dtype):
    if dtype in [mstype.float32, mstype.float16]:
        return True
    return False


class ClipByNorm(Cell):
    r"""
    Clips tensor values to a maximum :math:`L_2`-norm.

    The output of this layer remains the same if the :math:`L_2`-norm of the input tensor
    is not greater than the argument clip_norm. Otherwise the tensor will be normalized as:

    .. math::
        \text{output}(X) = \frac{\text{clip_norm} * X}{L_2(X)},

    where :math:`L_2(X)` is the :math:`L_2`-norm of :math:`X`.

    Args:
        axis (Union[None, int, tuple(int)): Compute the L2-norm along the Specific dimension.
                                            Default: None, all dimensions to calculate.

    Inputs:
        - **input** (Tensor) - Tensor of shape N-D. The type must be float32 or float16.
        - **clip_norm** (Tensor) - A scalar Tensor of shape :math:`()` or :math:`(1)`.
                                   Or a tensor shape can be broadcast to input shape.

    Outputs:
        Tensor, clipped tensor with the same shape as the input, whose type is float32.

    Examples:
        >>> net = nn.ClipByNorm()
        >>> input = Tensor(np.random.randint(0, 10, [4, 16]), mindspore.float32)
        >>> clip_norm = Tensor(np.array([100]).astype(np.float32))
        >>> net(input, clip_norm).shape
        (4, 16)

    """

    def __init__(self, axis=None):
        super(ClipByNorm, self).__init__()
        if axis is None:
            axis = ()
        if isinstance(axis, tuple):
            for idx, item in enumerate(axis):
                Validator.check_value_type("axis[%d]" % idx, item, [int], self.cls_name)
        self.axis = Validator.check_value_type('axis', axis, [int, tuple], self.cls_name)
        self.reduce_sum = P.ReduceSum(keep_dims=True)
        self.select_ = P.Select()
        self.greater_ = P.Greater()
        self.cast = P.Cast()
        self.sqrt = P.Sqrt()
        self.max_op = P.Maximum()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.fill = P.Fill()
        self.expand_dims = P.ExpandDims()
        self.dtype = P.DType()

    @ms_function
    def construct(self, x, clip_norm):
        """add ms_function decorator for pynative mode"""
        mul_x = F.square(x)
        l2sum = self.cast(self.reduce_sum(mul_x, self.axis), mstype.float32)
        cond = self.greater_(l2sum, 0)
        ones_ = self.fill(self.dtype(cond), self.shape(cond), 1.0)
        l2sum_safe = self.select_(cond, l2sum, self.cast(ones_, self.dtype(l2sum)))
        l2norm = self.select_(cond, self.sqrt(l2sum_safe), l2sum)

        _dtype_check(self.dtype(x))
        if _is_equal_one(clip_norm):
            intermediate = x
        else:
            intermediate = x * clip_norm

        max_norm = self.max_op(l2norm, clip_norm)
        if self.axis is None:
            max_norm = self.expand_dims(max_norm, -1)
        values_clip = self.cast(intermediate, mstype.float32) / max_norm
        values_clip = self.reshape(values_clip, self.shape(x))
        values_clip = identity(values_clip)
        return values_clip


class Norm(Cell):
    """
    Computes the norm of vectors, currently including Euclidean norm, i.e., :math:`L_2`-norm.

    Args:
        axis (Union[tuple, int]): The axis over which to compute vector norms. Default: ().
        keep_dims (bool): If true, the axis indicated in `axis` are kept with size 1. Otherwise,
                   the dimensions in `axis` are removed from the output shape. Default: False.

    Inputs:
        - **input** (Tensor) - Tensor which is not empty.

    Outputs:
        Tensor, output tensor with dimensions in 'axis' reduced to 1 will be returned if 'keep_dims' is True;
        otherwise a Tensor with dimensions in 'axis' removed is returned.

    Examples:
        >>> net = nn.Norm(axis=0)
        >>> input = Tensor(np.random.randint(0, 10, [2, 4]), mindspore.float32)
        >>> net(input)
        [2.236068 9.848858 4. 5.656854]
    """

    def __init__(self, axis=(), keep_dims=False):
        super(Norm, self).__init__()
        Validator.check_value_type("keep_dims", keep_dims, [bool], self.cls_name)
        self.axis = axis
        self.keep_dims = keep_dims
        self.reduce_sum = P.ReduceSum(True)
        self.sqrt = P.Sqrt()
        self.squeeze = P.Squeeze(self.axis)

    def construct(self, x):
        x = self.sqrt(self.reduce_sum(F.square(x), self.axis))

        if not self.keep_dims:
            x = self.squeeze(x)
        return x

    def extend_repr(self):
        return 'axis={}, keep_dims={}'.format(self.axis, self.keep_dims)


class OneHot(Cell):
    """
    Returns a one-hot tensor.

    The locations represented by indices in argument 'indices' take value on_value,
    while all other locations take value off_value.

    Note:
        If the input indices is rank :math:`N`, the output will have rank :math:`N+1`. The new
        axis is created at dimension `axis`.

    Args:
        axis (int): Features x depth if axis is -1, depth x features
                    if axis is 0. Default: -1.
        depth (int): A scalar defining the depth of the one hot dimension. Default: 1.
        on_value (float): A scalar defining the value to fill in output[i][j]
                          when indices[j] = i. Default: 1.0.
        off_value (float): A scalar defining the value to fill in output[i][j]
                           when indices[j] != i. Default: 0.0.
        dtype (:class:`mindspore.dtype`): Data type of 'on_value' and 'off_value', not the
                                          data type of indices. Default: mindspore.float32.

    Inputs:
        - **indices** (Tensor) - A tensor of indices of data type mindspore.int32 and arbitrary shape.

    Outputs:
        Tensor, the one-hot tensor of data type 'dtype' with dimension at 'axis' expanded to 'depth' and filled with
        on_value and off_value.

    Examples:
        >>> net = nn.OneHot(depth=4, axis=1)
        >>> indices = Tensor([[1, 3], [0, 2]], dtype=mindspore.int32)
        >>> net(indices)
        [[[0. 0.]
          [1. 0.]
          [0. 0.]
          [0. 1.]]
         [[1. 0.]
          [0. 0.]
          [0. 1.]
          [0. 0.]]]
    """

    def __init__(self, axis=-1, depth=1, on_value=1.0, off_value=0.0, dtype=mstype.float32):
        super(OneHot, self).__init__()
        self.onehot = P.OneHot(axis)
        self.depth = depth
        self.dtype = dtype
        self.on_value = on_value
        self.off_value = off_value

    def construct(self, indices):
        return self.onehot(indices, self.depth, F.cast(self.on_value, self.dtype), F.cast(self.off_value, self.dtype))



class Pad(Cell):
    """
    Pads the input tensor according to the paddings and mode.

    Args:
        paddings (tuple): The shape of parameter `paddings` is (N, 2). N is the rank of input data. All elements of
            paddings are int type. For `D` th dimension of input, paddings[D, 0] indicates how many sizes to be
            extended ahead of the `D` th dimension of the input tensor, and paddings[D, 1] indicates how many sizes to
            be extended behind of the `D` th dimension of the input tensor.
        mode (str): Specifies padding mode. The optional values are "CONSTANT", "REFLECT", "SYMMETRIC".
            Default: "CONSTANT".

    Inputs:
        - **input_x** (Tensor) - The input tensor.

    Outputs:
        Tensor, the tensor after padding.

        - If `mode` is "CONSTANT", it fills the edge with 0, regardless of the values of the `input_x`.
          If the `input_x` is [[1,2,3],[4,5,6],[7,8,9]] and `paddings` is [[1,1],[2,2]], then the
          Outputs is [[0,0,0,0,0,0,0],[0,0,1,2,3,0,0],[0,0,4,5,6,0,0],[0,0,7,8,9,0,0],[0,0,0,0,0,0,0]].
        - If `mode` is "REFLECT", it uses a way of symmetrical copying throught the axis of symmetry to fill in.
          If the `input_x` is [[1,2,3],[4,5,6],[7,8,9]] and `paddings` is [[1,1],[2,2]], then the
          Outputs is [[6,5,4,5,6,5,4],[3,2,1,2,3,2,1],[6,5,4,5,6,5,4],[9,8,7,8,9,8,7],[6,5,4,5,6,5,4]].
        - If `mode` is "SYMMETRIC", the filling method is similar to the "REFLECT". It is also copied
          according to the symmetry axis, except that it includes the symmetry axis. If the `input_x`
          is [[1,2,3],[4,5,6],[7,8,9]] and `paddings` is [[1,1],[2,2]], then the Outputs is
          [[2,1,1,2,3,3,2],[2,1,1,2,3,3,2],[5,4,4,5,6,6,5],[8,7,7,8,9,9,8],[8,7,7,8,9,9,8]].

    Examples:
        >>> from mindspore import Tensor
        >>> from mindspore.ops import operations as P
        >>> import mindspore.nn as nn
        >>> import numpy as np
        >>> class Net(nn.Cell):
        >>>     def __init__(self):
        >>>         super(Net, self).__init__()
        >>>         self.pad = nn.Pad(paddings=((1,1),(2,2)), mode="CONSTANT")
        >>>     def construct(self, x):
        >>>         return self.pad(x)
        >>> x = np.random.random(size=(2, 3)).astype(np.float32)
        >>> pad = Net()
        >>> ms_output = pad(Tensor(x))
    """

    def __init__(self, paddings, mode="CONSTANT"):
        super(Pad, self).__init__()
        self.mode = mode
        self.paddings = paddings
        Validator.check_string(self.mode, ["CONSTANT", "REFLECT", "SYMMETRIC"], 'mode', self.cls_name)
        if not isinstance(paddings, tuple):
            raise TypeError('Paddings must be tuple type.')
        for item in paddings:
            if len(item) != 2:
                raise ValueError('The shape of paddings must be (n, 2).')
        if len(paddings) > 4:
            raise ValueError('Only padding up to 4 dims is supported')
        if mode == "CONSTANT":
            self.pad = P.Pad(self.paddings)
        else:
            self.paddings = Tensor(np.array(self.paddings))
            self.pad = P.MirrorPad(mode=mode)

    def construct(self, x):
        if self.mode == "CONSTANT":
            x = self.pad(x)
        else:
            x = self.pad(x, self.paddings)
        return x


class Unfold(Cell):
    """
    Extract patches from images.
    The input tensor must be a 4-D tensor and the data format is NCHW.

    Args:
        ksizes (Union[tuple[int], list[int]]): The size of sliding window, must be a tuple or a list of integers,
            and the format is [1, ksize_row, ksize_col, 1].
        strides (Union[tuple[int], list[int]]): Distance between the centers of the two consecutive patches,
            must be a tuple or list of int, and the format is [1, stride_row, stride_col, 1].
        rates (Union[tuple[int], list[int]]): In each extracted patch, the gap between the corresponding dimension
            pixel positions, must be a tuple or a list of integers, and the format is [1, rate_row, rate_col, 1].
        padding (str): The type of padding algorithm, is a string whose value is "same" or "valid",
            not case sensitive. Default: "valid".

            - same: Means that the patch can take the part beyond the original image, and this part is filled with 0.

            - valid: Means that the taken patch area must be completely covered in the original image.

    Inputs:
        - **input_x** (Tensor) - A 4-D tensor whose shape is [in_batch, in_depth, in_row, in_col] and
          data type is number.

    Outputs:
        Tensor, a 4-D tensor whose data type is same as 'input_x',
        and the shape is [out_batch, out_depth, out_row, out_col], the out_batch is the same as the in_batch.

    Examples:
        >>> net = Unfold(ksizes=[1, 2, 2, 1], strides=[1, 2, 2, 1], rates=[1, 2, 2, 1])
        >>> image = Tensor(np.ones([2, 3, 6, 6]), dtype=mstype.float16)
        >>> net(image)
        Tensor ([[[[1, 1] [1, 1]] [[1, 1], [1, 1]] [[1, 1] [1, 1]], [[1, 1] [1, 1]], [[1, 1] [1, 1]],
                [[1, 1], [1, 1]]]], shape=(2, 12, 2, 2), dtype=mstype.float16)
    """

    def __init__(self, ksizes, strides, rates, padding="valid"):
        super(Unfold, self).__init__()
        self.extract_image_patches = inner.ExtractImagePatches(ksizes, strides, rates, padding)
        self.transpose = P.Transpose()
        self.format_NHWC = (0, 2, 3, 1)
        self.format_NCHW = (0, 3, 1, 2)
        self.is_ge = context.get_context("enable_ge")

    def construct(self, input_x):
        if self.is_ge:
            x_transpose = self.transpose(input_x, self.format_NHWC)
            ret = self.extract_image_patches(x_transpose)
            result = self.transpose(ret, self.format_NCHW)
        else:
            result = self.extract_image_patches(input_x)
        return result


@constexpr
def _get_matrix_diag_assist(x_shape, x_dtype):
    Validator.check_int(len(x_shape), 1, Rel.GE, "x rank", "_get_matrix_diag_assist")
    base_eye = np.eye(x_shape[-1], x_shape[-1]).reshape(-1)
    assist = np.tile(base_eye, x_shape[:-1]).reshape(x_shape + (x_shape[-1],))
    return Tensor(assist, x_dtype)


@constexpr
def _get_matrix_diag_part_assist(x_shape, x_dtype):
    Validator.check_int(len(x_shape), 2, Rel.GE, "x rank", "_get_matrix_diag_part_assist")
    base_eye = np.eye(x_shape[-2], x_shape[-1]).reshape(-1)
    assist = np.tile(base_eye, x_shape[:-2]).reshape(x_shape)
    return Tensor(assist, x_dtype)


class MatrixDiag(Cell):
    """
    Returns a batched diagonal tensor with a given batched diagonal values.

    Inputs:
        - **x** (Tensor) - The diagonal values. It can be one of the following data types:
          float32, float16, int32, int8, and uint8.

    Outputs:
        Tensor, has the same type as input `x`. The shape must be x.shape + (x.shape[-1], ).

    Examples:
        >>> x = Tensor(np.array([1, -1]), mstype.float32)
        >>> matrix_diag = nn.MatrixDiag()
        >>> result = matrix_diag(x)
        [[1.   0.]
         [0.  -1.]]
    """
    def __init__(self):
        super(MatrixDiag, self).__init__()
        self.matrix_diag = inner.MatrixDiag()
        self.dtype = P.DType()

    def construct(self, input_x):
        x_shape = F.shape(input_x)
        x_dtype = self.dtype(input_x)
        assist = _get_matrix_diag_assist(x_shape, x_dtype)
        out_matrix_diag = self.matrix_diag(input_x, assist)
        return out_matrix_diag


class MatrixDiagPart(Cell):
    r"""
    Returns the batched diagonal part of a batched tensor.

    Inputs:
        - **x** (Tensor) - The batched tensor. It can be one of the following data types:
          float32, float16, int32, int8, and uint8.

    Outputs:
        Tensor, has the same type as input `x`. The shape must be x.shape[:-2] + [min(x.shape[-2:])].

    Examples:
        >>> x = Tensor([[[-1, 0], [0, 1]], [[-1, 0], [0, 1]], [[-1, 0], [0, 1]]], mindspore.float32)
        >>> matrix_diag_part = nn.MatrixDiagPart()
        >>> result = matrix_diag_part(x)
        [[-1., 1.], [-1., 1.], [-1., 1.]]
    """
    def __init__(self):
        super(MatrixDiagPart, self).__init__()
        self.matrix_diag_part = inner.MatrixDiagPart()
        self.dtype = P.DType()

    def construct(self, input_x):
        x_shape = F.shape(input_x)
        x_dtype = self.dtype(input_x)
        assist = _get_matrix_diag_part_assist(x_shape, x_dtype)
        out_matrix_diag_part = self.matrix_diag_part(input_x, assist)
        return out_matrix_diag_part


class MatrixSetDiag(Cell):
    r"""
    Modify the batched diagonal part of a batched tensor.

    Inputs:
        - **x** (Tensor) - The batched tensor. Rank k+1, where k >= 1. It can be one of the following data types:
          float32, float16, int32, int8, and uint8.
        - **diagonal** (Tensor) - The diagonal values. Must have the same type as input `x`. Rank k, where k >= 1.

    Outputs:
        Tensor, has the same type and shape as input `x`.

    Examples:
        >>> x = Tensor([[[-1, 0], [0, 1]], [[-1, 0], [0, 1]], [[-1, 0], [0, 1]]], mindspore.float32)
        >>> diagonal = Tensor([[-1., 2.], [-1., 1.], [-1., 1.]], mindspore.float32)
        >>> matrix_set_diag = nn.MatrixSetDiag()
        >>> result = matrix_set_diag(x, diagonal)
        [[[-1, 0], [0, 2]], [[-1, 0], [0, 1]], [[-1, 0], [0, 1]]]
    """
    def __init__(self):
        super(MatrixSetDiag, self).__init__()
        self.matrix_set_diag = inner.MatrixSetDiag()
        self.dtype = P.DType()

    def construct(self, input_x, diagonal):
        x_shape = F.shape(input_x)
        x_dtype = self.dtype(input_x)
        assist = _get_matrix_diag_part_assist(x_shape, x_dtype)
        out_matrix_set_diag = self.matrix_set_diag(input_x, diagonal, assist)
        return out_matrix_set_diag

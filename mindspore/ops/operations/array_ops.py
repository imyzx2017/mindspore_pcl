# coding: utf-8

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

"""Operators for array."""

import copy
import functools
import itertools
import numbers

import numpy as np

from .._utils import get_concat_offset
from ..operations.math_ops import _infer_shape_reduce
from ..primitive import Primitive, PrimitiveWithInfer, PrimitiveWithCheck, prim_attr_register, _run_op
from .. import signature as sig
from ..._checkparam import Rel
from ..._checkparam import Validator as validator
from ...common import dtype as mstype
from ...common.parameter import Parameter
from ...common.tensor import Tensor


class _ScatterOp(PrimitiveWithInfer):
    """
    Defines Scatter operators
    """
    __mindspore_signature__ = (
        sig.make_sig('x', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('indices', dtype=sig.sig_dtype.T1),
        sig.make_sig('updates', dtype=sig.sig_dtype.T)
    )

    def _check_scatter_shape(self, x_shape, indices_shape, updates_shape, prim_name):
        if indices_shape != [-1] and updates_shape and updates_shape != indices_shape + x_shape[1:]:
            raise ValueError(f"For '{prim_name}', "
                             f"updates_shape = indices_shape + x_shape[1:], but got x_shape: {x_shape}, "
                             f"indices_shape: {indices_shape}, updates_shape: {updates_shape}.")

    @prim_attr_register
    def __init__(self, use_locking=False):
        """Initialize _ScatterOp"""
        validator.check_value_type('use_locking', use_locking, [bool], self.name)
        self.init_prim_io_names(inputs=['x', 'indices', 'updates'], outputs=['y'])

    def infer_shape(self, x_shape, indices_shape, updates_shape):
        self._check_scatter_shape(x_shape, indices_shape, updates_shape, self.name)
        return x_shape

    def infer_dtype(self, x_dtype, indices_dtype, updates_dtype):
        validator.check_tensor_dtype_valid('indices', indices_dtype, [mstype.int32], self.name)
        args = {"x": x_dtype, "updates": updates_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.number_type, self.name)
        return x_dtype


class _ScatterOp_Dynamic(PrimitiveWithCheck):
    """
    Defines Scatter operators with dynamic shape
    """
    __mindspore_signature__ = (
        sig.make_sig('x', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('indices', dtype=sig.sig_dtype.T1),
        sig.make_sig('updates', dtype=sig.sig_dtype.T)
    )

    def _check_scatter_shape(self, x_shape, indices_shape, updates_shape, prim_name):
        if indices_shape != [-1] and updates_shape and updates_shape != indices_shape + x_shape[1:]:
            raise ValueError(f"For '{prim_name}', "
                             f"updates_shape = indices_shape + x_shape[1:], but got x_shape: {x_shape}, "
                             f"indices_shape: {indices_shape}, updates_shape: {updates_shape}.")

    @prim_attr_register
    def __init__(self, use_locking=False):
        """Initialize _ScatterOp_Dynamic"""
        validator.check_value_type('use_locking', use_locking, [bool], self.name)
        self.init_prim_io_names(inputs=['x', 'indices', 'updates'], outputs=['y'])

    def check_shape(self, x_shape, indices_shape, updates_shape):
        self._check_scatter_shape(x_shape, indices_shape, updates_shape, self.name)

    def check_dtype(self, x_dtype, indices_dtype, updates_dtype):
        validator.check_tensor_dtype_valid('indices', indices_dtype, [mstype.int32], self.name)
        args = {"x": x_dtype, "updates": updates_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.number_type, self.name)


class _ScatterNdOp(_ScatterOp):
    """
    Defines _ScatterNd operators
    """

    def _check_scatter_shape(self, x_shape, indices_shape, updates_shape, prim_name):
        validator.check('the dimension of x', len(x_shape),
                        'the dimension of indices', indices_shape[-1], Rel.GE)
        if indices_shape[:-1] + x_shape[indices_shape[-1]:] != updates_shape:
            raise ValueError(f"For '{prim_name}', updates_shape = "
                             f"indices_shape[:-1] + x_shape[indices_shape[-1]:], but got x_shape: {x_shape}, "
                             f"indices_shape: {indices_shape}, updates_shape: {updates_shape}.")


def _check_infer_attr_reduce(axis, keep_dims, prim_name):
    validator.check_value_type('keep_dims', keep_dims, [bool], prim_name)
    validator.check_value_type('axis', axis, [int, tuple], prim_name)
    if isinstance(axis, tuple):
        for index, value in enumerate(axis):
            validator.check_value_type('axis[%d]' % index, value, [int], prim_name)


class ExpandDims(PrimitiveWithInfer):
    """
    Adds an additional dimension at the given axis.

    Note:
        If the specified axis is a negative number, the index is counted
        backward from the end and starts at 1.

    Raises:
        ValueError: If axis is not an integer or not in the valid range.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        - **axis** (int) - Specifies the dimension index at which to expand
          the shape of `input_x`. The value of axis must be in the range
          `[-input_x.dim()-1, input_x.dim()]`. Only constant value is allowed.

    Outputs:
        Tensor, the shape of tensor is :math:`(1, x_1, x_2, ..., x_R)` if the
        value of `axis` is 0.

    Examples:
        >>> input_tensor = Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32)
        >>> expand_dims = P.ExpandDims()
        >>> output = expand_dims(input_tensor, 0)
        >>> print(output)
        [[[2.0, 2.0],
          [2.0, 2.0]]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize ExpandDims"""
        self.init_prim_io_names(inputs=['x', 'axis'], outputs=['output'])

    def __infer__(self, x, axis):
        validator.check_subclass("input_x", x['dtype'], mstype.tensor, self.name)
        x_shape = list(x['shape'])
        axis_v = axis['value']
        rank = len(x_shape)
        validator.check_int_range(axis_v, -rank - 1, rank, Rel.INC_BOTH, 'axis', self.name)
        value = None
        if x['value'] is not None:
            value = x['value'].asnumpy()
            value = np.expand_dims(value, axis_v)
            value = Tensor(value)
        if axis_v < 0:
            axis_v = rank + 1 + axis_v
        x_shape.insert(axis_v, 1)
        out = {'shape': x_shape,
               'dtype': x['dtype'],
               'value': value}
        if 'min_shape' in x and 'max_shape' in x:
            out['min_shape'] = x['min_shape']
            out['min_shape'].insert(axis_v, 1)
            out['max_shape'] = x['max_shape']
            out['max_shape'].insert(axis_v, 1)
        return out


class DType(PrimitiveWithInfer):
    """
    Returns the data type of input tensor as mindspore.dtype.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        mindspore.dtype, the data type of a tensor.

    Examples:
        >>> input_tensor = Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32)
        >>> output = P.DType()(input_tensor)
        >>> print(output)
        Float32
    """

    @prim_attr_register
    def __init__(self):
        """Initialize DType"""

    def __infer__(self, x):
        validator.check_subclass("input_x", x['dtype'], mstype.tensor, self.name)
        out = {'shape': (),
               'dtype': mstype.type_type,
               'value': x['dtype'].element_type()}
        return out


class SameTypeShape(PrimitiveWithInfer):
    """
    Checks whether data type and shape of two tensors are the same.

    Raises:
        TypeError: If the data types of two tensors are not the same.
        ValueError: If the shapes of two tensors are not the same.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        - **input_y** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_S)`.

    Outputs:
        Tensor, the shape of tensor is :math:`(x_1, x_2, ..., x_R)`,
        if data type and shape of `input_x` and `input_y` are the same.

    Examples:
        >>> input_x = Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32)
        >>> input_y = Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32)
        >>> out = P.SameTypeShape()(input_x, input_y)
        >>> print(out)
        [[2. 2.]
         [2. 2.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Same"""

    def __call__(self, x, y):
        """run in PyNative mode"""
        validator.check_value_type('x', x, Tensor, self.name)
        validator.check_value_type('y', y, Tensor, self.name)
        validator.check('x dtype', x.dtype, 'y dtype', y.dtype, Rel.EQ, self.name, TypeError)
        validator.check('x shape', x.shape, 'y shape', y.shape, Rel.EQ, self.name)
        return x

    def __infer__(self, x, y):
        validator.check_subclass('x', x['dtype'], mstype.tensor, self.name)
        validator.check_subclass('y', y['dtype'], mstype.tensor, self.name)
        validator.check('x dtype', x['dtype'], 'y dtype', y['dtype'], Rel.EQ, self.name, TypeError)
        validator.check('x shape', x['shape'], 'y shape', y['shape'], Rel.EQ, self.name)
        return x


class Cast(PrimitiveWithInfer):
    """
    Returns a tensor with the new specified data type.

    Inputs:
        - **input_x** (Union[Tensor, Number]) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
          The tensor to be cast.
        - **type** (dtype.Number) - The valid data type of the output tensor. Only constant value is allowed.

    Outputs:
        Tensor, the shape of tensor is the same as `input_x`, :math:`(x_1, x_2, ..., x_R)`.

    Examples:
        >>> input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
        >>> input_x = Tensor(input_np)
        >>> type_dst = mindspore.float16
        >>> cast = P.Cast()
        >>> output = cast(input_x, type_dst)
        >>> print(output.dtype)
        Float16
        >>> print(output.shape)
        (2, 3, 4, 5)
    """

    @prim_attr_register
    def __init__(self):
        # if primitive need setattr in __infer__ need add this flag
        """Initialize Cast"""
        self.init_prim_io_names(inputs=['x', 'dst_type'], outputs=['output'])

    def check_elim(self, x, dtype):
        if isinstance(x, (Tensor, numbers.Number, Parameter)):
            if isinstance(x, Tensor) and x.dtype == dtype:
                return (True, x)
            if isinstance(x, numbers.Number):
                return (True, Tensor(x, dtype=dtype))
            if isinstance(x, Parameter):
                data = x.data
                if data.dtype == dtype:
                    return (True, x)
        return (False, None)

    def __infer__(self, x, t):
        src_type = x['dtype']
        dst_type = t['value']

        validator.check_subclass("input_x", src_type, [mstype.tensor, mstype.number], self.name)
        validator.check_subclass("type", dst_type, mstype.number, self.name)

        if isinstance(src_type, type(mstype.tensor)):
            src_type = x['dtype'].element_type()
        if isinstance(dst_type, type(mstype.tensor)):
            dst_type = dst_type.element_type()
        self.add_prim_attr('DstT', dst_type)
        self.add_prim_attr('SrcT', src_type)
        self.add_prim_attr('dst_type', dst_type)

        value = None
        if x['value'] is not None:
            np_dst_type = mstype.dtype_to_nptype(dst_type)
            if isinstance(x['value'], (int, float)):
                value = Tensor(np.array(x['value']).astype(np_dst_type))
            else:
                value = Tensor(x['value'].asnumpy().astype(np_dst_type))

        out = {'shape': x['shape'],
               'dtype': mstype.tensor_type(t['value']),
               'value': value}
        if 'min_shape' in x and 'max_shape' in x:
            out['min_shape'] = x['min_shape']
            out['max_shape'] = x['max_shape']
        return out


class IsSubClass(PrimitiveWithInfer):
    """
    Checks whether one type is subtraction class of another type.

    Inputs:
        - **sub_type** (mindspore.dtype) - The type to be checked. Only constant value is allowed.
        - **type_** (mindspore.dtype) - The target type. Only constant value is allowed.

    Outputs:
        bool, the check result.

    Examples:
        >>> result = P.IsSubClass()(mindspore.int32,  mindspore.intc)
        >>> print(result)
        True
    """

    @prim_attr_register
    def __init__(self):
        pass

    def __infer__(self, sub_type, type_):
        sub_type_t = sub_type['value']
        type_v = type_['value']

        validator.check_value_type("sub_type", sub_type_t, [mstype.Type], self.name)
        validator.check_value_type("type_", type_v, [mstype.Type], self.name)

        value = mstype.issubclass_(sub_type_t, type_v)

        out = {'shape': (),
               'dtype': mstype.type_type,
               'value': value}
        return out


class IsInstance(PrimitiveWithInfer):
    """
    Checks whether an object is an instance of a target type.

    Inputs:
        - **inst** (Any Object) - The instance to be checked. Only constant value is allowed.
        - **type_** (mindspore.dtype) - The target type. Only constant value is allowed.

    Outputs:
        bool, the check result.

    Examples:
        >>> a = 1
        >>> result = P.IsInstance()(a, mindspore.int64)
        >>> print(result)
        True
    """

    @prim_attr_register
    def __init__(self):
        pass

    def __infer__(self, inst, type_):
        sub_type_t = inst['dtype']
        type_v = type_['value']

        validator.check_value_type("type_", type_v, [mstype.Type], self.name)

        if type_v == mstype.list_:
            value = isinstance(sub_type_t, list)
        elif type_v == mstype.tuple_:
            value = isinstance(sub_type_t, tuple)
        else:
            value = mstype.issubclass_(sub_type_t, type_v)

        out = {'shape': (),
               'dtype': mstype.type_type,
               'value': value}
        return out


class Reshape(PrimitiveWithInfer):
    """
    Reshapes input tensor with the same values based on a given shape tuple.

    Raises:
        ValueError: Given a shape tuple, if it has several -1; or if the product
            of its elements is less than or equal to 0 or cannot be divided by the product
            of the input tensor shape; or if it does not match the input's array size.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        - **input_shape** (tuple[int]) - The input tuple is constructed by multiple
          integers, i.e., :math:`(y_1, y_2, ..., y_S)`. Only constant value is allowed.

    Outputs:
        Tensor, the shape of tensor is :math:`(y_1, y_2, ..., y_S)`.

    Examples:
        >>> input_tensor = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
        >>> reshape = P.Reshape()
        >>> output = reshape(input_tensor, (3, 2))
        >>> print(output)
        [[-0.1 0.3]
         [3.6 0.4 ]
         [0.5 -3.2]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Reshape"""
        self.init_prim_io_names(inputs=['tensor', 'shape'], outputs=['output'])

    def __infer__(self, x, shape):
        shape_v = shape['value']
        x_shp = x['shape']
        validator.check_subclass("x", x['dtype'], mstype.tensor, self.name)
        validator.check_value_type("shape", shape_v, [tuple], self.name)
        shape_v = list(shape_v)
        if 'max_shape' in x:
            x_max_shape = x['max_shape']
        else:
            x_max_shape = x['shape']
        if 'min_shape' in x:
            x_min_shape = x['min_shape']
        else:
            x_min_shape = x['shape']
        neg_index = -1
        dim_prod = 1
        for i, shp_i in enumerate(shape_v):
            validator.check_value_type("shape[%d]" % i, shp_i, [int], self.name)
            if shp_i == -1:
                if neg_index != -1:
                    raise ValueError(f'The shape can only has one -1 at most, but {shape_v}.')
                neg_index = i
            else:
                dim_prod *= shp_i
        arr_prod = np.prod(x_shp)
        max_arr_prod = np.prod(x_max_shape)
        min_arr_prod = np.prod(x_min_shape)
        if dim_prod <= 0 or arr_prod % dim_prod != 0:
            raise ValueError(f'For \'{self.name}\' input_x\'s shape is {x_shp}, input_shape\'s value is {shape_v}.'
                             f'The product of input_x\'s shape should > 0, '
                             f'and can be divided by product of input_shape, '
                             f'but product of input_x\'s shape is {arr_prod}, product of input_shape is {dim_prod}.')
        max_shape = list(shape_v)
        min_shape = list(shape_v)
        if neg_index != -1:
            shape_v[neg_index] = int(arr_prod / dim_prod)
            max_shape[neg_index] = int(max_arr_prod / dim_prod)
            min_shape[neg_index] = int(min_arr_prod / dim_prod)
            dim_prod *= shape_v[neg_index]
        if dim_prod != arr_prod:
            raise ValueError(f'For \'{self.name}\' input_x\'s shape is {x_shp}, input_shape\'s value is {shape_v}.'
                             f'The product of input_x\'s shape should be equal to product of input_shape, '
                             f'but product of input_x\'s shape is {arr_prod}, product of input_shape is {dim_prod}.')

        value = None
        if x['value'] is not None:
            value = Tensor(x['value'].asnumpy().reshape(shape_v))

        out = {'shape': tuple(shape_v),
               'dtype': x['dtype'],
               'value': value,
               'max_shape': tuple(max_shape),
               'min_shape': tuple(min_shape)}
        return out


class Shape(PrimitiveWithInfer):
    """
    Returns the shape of input tensor.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        tuple[int], the output tuple is constructed by multiple integers,
        :math:`(x_1, x_2, ..., x_R)`.

    Examples:
        >>> input_tensor = Tensor(np.ones(shape=[3, 2, 1]), mindspore.float32)
        >>> shape = P.Shape()
        >>> output = shape(input_tensor)
        >>> print(output)
        (3, 2, 1)
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Shape"""

    def __infer__(self, x):
        validator.check_subclass("input_x", x['dtype'], mstype.tensor, self.name)
        out = {'shape': (),
               'dtype': mstype.tuple_,
               'value': tuple(x['shape'])}
        return out


class DynamicShape(Primitive):
    """
    Returns the shape of input tensor.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        Tensor[int], 1-dim Tensor of type int32

    Examples:
        >>> input_tensor = Tensor(np.ones(shape=[3, 2, 1]), mindspore.float32)
        >>> shape = P.DynamicShape()
        >>> output = shape(input_tensor)
    """

    @prim_attr_register
    def __init__(self):
        """init Shape"""
        self.init_prim_io_names(inputs=['tensor'], outputs=['output'])
        self.add_prim_attr('is_dynamic_shape', True)
        self.add_prim_attr("dynamic_shape_depends", [0])


class Squeeze(PrimitiveWithInfer):
    """
    Returns a tensor with the same type but dimensions of 1 are removed based on `axis`.

    Note:
        The dimension index starts at 0 and must be in the range `[-input.dim(), input.dim())`.

    Raises:
        ValueError: If the corresponding dimension of the specified axis does not equal to 1.

    Args:
        axis (Union[int, tuple(int)]): Specifies the dimension indexes of shape to be removed, which will remove
            all the dimensions that are equal to 1. If specified, it must be int32 or int64.
            Default: (), an empty tuple.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        Tensor, the shape of tensor is :math:`(x_1, x_2, ..., x_S)`.

    Examples:
        >>> input_tensor = Tensor(np.ones(shape=[3, 2, 1]), mindspore.float32)
        >>> squeeze = P.Squeeze(2)
        >>> output = squeeze(input_tensor)
        >>> print(output)
        [[1. 1.]
         [1. 1.]
         [1. 1.]]
    """

    @prim_attr_register
    def __init__(self, axis=()):
        """Initialize Squeeze"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])
        validator.check_value_type('axis', axis, [int, tuple], self.name)
        if isinstance(axis, tuple):
            for idx, item in enumerate(axis):
                validator.check_value_type("axis[%d]" % idx, item, [int], self.name)
        else:
            self.axis = (axis,)
            self.add_prim_attr("axis", (axis,))

    def infer_shape(self, x_shape):
        axis = self.axis
        x_shape = list(x_shape)
        ndim = len(x_shape)
        if not axis:
            ret = [d for d in x_shape if d != 1]
        else:
            for a in axis:
                validator.check_int_range(a, -ndim, ndim - 1, Rel.INC_BOTH, 'axis or its elements', self.name)
                if x_shape[a] != 1:
                    raise ValueError('Cannot select an axis to squeeze out which has size not equal to one.')
            ret = [x_shape[i] for i in range(ndim) if not (i in axis or (i - ndim) in axis)]
        return ret

    def infer_dtype(self, x_dtype):
        validator.check_subclass("x", x_dtype, mstype.tensor, self.name)
        return x_dtype


class Transpose(PrimitiveWithCheck):
    """
    Permutes the dimensions of input tensor according to input permutation.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        - **input_perm** (tuple[int]) - The permutation to be converted. The input tuple is constructed by multiple
          indexes. The length of `input_perm` and the shape of `input_x` must be the same. Only constant value is
          allowed. Must be in the range [0, rank(input_x)).

    Outputs:
        Tensor, the type of output tensor is the same as `input_x` and the shape of output tensor is decided by the
        shape of `input_x` and the value of `input_perm`.

    Examples:
        >>> input_tensor = Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), mindspore.float32)
        >>> perm = (0, 2, 1)
        >>> transpose = P.Transpose()
        >>> output = transpose(input_tensor, perm)
        >>> print(output)
        [[[1. 4.]
          [2. 5.]
          [3. 6.]]
         [[7. 10.]
          [8. 11.]
          [9. 12.]]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Transpose"""
        self.init_prim_io_names(inputs=['x', 'perm'], outputs=['output'])

    def check_shape(self, x, perm):
        validator.check_value_type("perm", perm, [tuple], self.name)
        if len(x) != len(perm):
            raise ValueError('The dimension of x and perm must be equal.')

    def check_dtype(self, x, perm):
        validator.check_subclass("x", x, mstype.tensor, self.name)

class Unique(Primitive):
    """
    Returns the unique elements of input tensor and also return a tensor containing the index of each value of input
    tensor corresponding to the output unique tensor.

    Inputs:
        - **x** (Tensor) - The input tensor.

    Outputs:
        Tuple, containing Tensor objects `(y, idx)`, `y` is a tensor has the same type as `x`, `idx` is a tensor
        containing indices of elements in the input coressponding to the output tensor.

    Examples:
        >>> x = Tensor(np.array([1, 2, 5, 2]), mindspore.int32)
        >>> out = P.Unique()(x)
        (Tensor([1, 2, 5], mindspore.int32), Tensor([0, 1, 2, 1], mindspore.int32))
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x'], outputs=['output'])


class GatherV2(PrimitiveWithCheck):
    """
    Returns a slice of input tensor based on the specified indices and axis.

    Inputs:
        - **input_params** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
          The original Tensor.
        - **input_indices** (Tensor) - The shape of tensor is :math:`(y_1, y_2, ..., y_S)`.
          Specifies the indices of elements of the original Tensor. Must be in the range
          `[0, input_param.shape[axis])`.
        - **axis** (int) - Specifies the dimension index to gather indices.

    Outputs:
        Tensor, the shape of tensor is :math:`(z_1, z_2, ..., z_N)`.

    Examples:
        >>> input_params = Tensor(np.array([[1, 2, 7, 42], [3, 4, 54, 22], [2, 2, 55, 3]]), mindspore.float32)
        >>> input_indices = Tensor(np.array([1, 2]), mindspore.int32)
        >>> axis = 1
        >>> out = P.GatherV2()(input_params, input_indices, axis)
        >>> print(out)
        [[2.0, 7.0],
         [4.0, 54.0],
         [2.0, 55.0]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize index_select"""
        self.init_prim_io_names(inputs=['params', 'indices', 'axis'], outputs=['output'])
        self.add_prim_attr("dynamic_shape_depends", [2])

    def __check__(self, params, indices, axis):
        validator.check_subclass("params", params['dtype'], mstype.tensor, self.name)
        validator.check_tensor_dtype_valid("indices", indices['dtype'], mstype.int_type, self.name)
        validator.check_subclass("axis", axis['dtype'], mstype.int_, self.name)
        axis_v = axis['value']
        params_shp = params['shape']
        rank = len(params_shp)
        validator.check_int_range(axis_v, -rank, rank, Rel.INC_LEFT, "axis", self.name)

        if axis_v < 0:
            axis_v += rank
        out_shape = params_shp[:axis_v] + indices['shape'] + params_shp[axis_v + 1:]
        out = {'shape': out_shape,
               'dtype': params['dtype'],
               'value': None}
        if 'min_shape' in indices and 'max_shape' in indices:
            out['min_shape'] = params_shp[:axis_v] + indices['min_shape'] + params_shp[axis_v + 1:]
            out['max_shape'] = params_shp[:axis_v] + indices['max_shape'] + params_shp[axis_v + 1:]
        return out


class SparseGatherV2(GatherV2):
    """
    Returns a slice of input tensor based on the specified indices and axis.

    Inputs:
        - **input_params** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
          The original Tensor.
        - **input_indices** (Tensor) - The shape of tensor is :math:`(y_1, y_2, ..., y_S)`.
          Specifies the indices of elements of the original Tensor, must be in the range
          `[0, input_param.shape[axis])`.
        - **axis** (int) - Specifies the dimension index to gather indices.

    Outputs:
        Tensor, the shape of tensor is :math:`(z_1, z_2, ..., z_N)`.

    Examples:
        >>> input_params = Tensor(np.array([[1, 2, 7, 42], [3, 4, 54, 22], [2, 2, 55, 3]]), mindspore.float32)
        >>> input_indices = Tensor(np.array([1, 2]), mindspore.int32)
        >>> axis = 1
        >>> out = P.SparseGatherV2()(input_params, input_indices, axis)
    """


class Padding(PrimitiveWithInfer):
    """
    Extends the last dimension of input tensor from 1 to pad_dim_size, by filling with 0.

    Args:
        pad_dim_size (int): The value of the last dimension of x to be extended, which must be positive.

    Inputs:
        - **x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`. The rank of x must be at least 2.
          The last dimension of x must be 1.

    Outputs:
        Tensor, the shape of tensor is :math:`(z_1, z_2, ..., z_N)`.

    Examples:
        >>> x = Tensor(np.array([[8], [10]]), mindspore.float32)
        >>> pad_dim_size = 4
        >>> out = P.Padding(pad_dim_size)(x)
        >>> print(out)
        [[8, 0, 0, 0], [10, 0, 0, 0]]
    """

    @prim_attr_register
    def __init__(self, pad_dim_size=8):
        """Initialize padding"""
        validator.check_value_type("pad_dim_size", pad_dim_size, [int], self.name)
        validator.check_positive_int(pad_dim_size, "pad_dim_size", self.name)
        self.pad_dim_size = pad_dim_size

    def __infer__(self, x):
        validator.check_subclass("x", x['dtype'], mstype.tensor, self.name)
        x_shape = list(x['shape'])
        validator.check_int(len(x_shape), 1, Rel.GT, "rank of x", self.name)
        validator.check_int(x_shape[-1], 1, Rel.EQ, "last dim of x", self.name)
        out_shape = x_shape
        out_shape[-1] = self.pad_dim_size
        out = {'shape': out_shape,
               'dtype': x['dtype'],
               'value': None}
        return out


class UniqueWithPad(PrimitiveWithInfer):
    """
    Return unique elements and relative indexes in 1-D tensor, fill with pad num.

    Inputs:
        - **x** (Tensor) - The tensor need to be unique. Must be 1-D vector with types: int32, int64.
        - **pad_num** (int) - Pad num.

    Outputs:
        tuple(Tensor), tuple of 2 tensors, y and idx.
        - y (Tensor) - The unique elements filled with pad_num, the shape and type same as x.
        - idx (Tensor) - The index of each value of x in the unique output y, the shape and type same as x.

    Examples:
        >>> x = Tensor(np.array([1, 1, 5, 5, 4, 4, 3, 3, 2, 2,]), mindspore.int32)
        >>> pad_num = 8
        >>> out = P.UniqueWithPad()(x, pad_num)
        >>> print(out)
        ([1, 5, 4, 3, 2, 8, 8, 8, 8, 8], [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    """

    @prim_attr_register
    def __init__(self):
        """init UniqueWithPad"""

    def __infer__(self, x, pad_num):
        validator.check_tensor_dtype_valid("x", x['dtype'], [mstype.int32, mstype.int64], self.name)
        validator.check_subclass("pad_num", pad_num['dtype'], [mstype.int32, mstype.int64], self.name)
        x_shape = list(x['shape'])
        validator.check("rank of x", len(x_shape), "expected", 1, Rel.EQ, self.name)
        out_shape = x_shape
        out = {'shape': (out_shape, out_shape),
               'dtype': (x['dtype'], x['dtype']),
               'value': None}
        return out


class Split(PrimitiveWithInfer):
    """
    Splits input tensor into output_num of tensors along the given axis and output numbers.

    Args:
        axis (int): Index of the split position. Default: 0.
        output_num (int): The number of output tensors. Default: 1.

    Raises:
        ValueError: If `axis` is out of the range [-len(`input_x.shape`), len(`input_x.shape`)),
            or if the `output_num` is less than or equal to 0, or if the
            dimension which to split cannot be evenly divided by `output_num`.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        tuple[Tensor], the shape of each output tensor is the same, which is
        :math:`(y_1, y_2, ..., y_S)`.

    Examples:
        >>> split = P.Split(1, 2)
        >>> x = Tensor(np.array([[1, 1, 1, 1], [2, 2, 2, 2]]))
        >>> output = split(x)
        >>> print(output)
        ([[1, 1],
          [2, 2]],
         [[1, 1],
          [2, 2]])
    """

    @prim_attr_register
    def __init__(self, axis=0, output_num=1):
        """Initialize Split"""
        validator.check_value_type("axis", axis, [int], self.name)
        validator.check_value_type("output_num", output_num, [int], self.name)
        self.axis = axis
        self.output_num = output_num

    def __infer__(self, x):
        validator.check_subclass("x", x['dtype'], mstype.tensor, self.name)
        x_shape = list(x['shape'])
        dim = len(x_shape)
        validator.check_int_range(self.axis, -dim, dim, Rel.INC_LEFT, 'axis value', self.name)
        validator.check_positive_int(self.output_num, "output_num", self.name)
        output_valid_check = x_shape[self.axis] % self.output_num
        if output_valid_check != 0:
            raise ValueError(f"x_shape[{self.axis}] {x_shape[self.axis]} must be divide exactly by"
                             f" output_num {self.output_num}")

        x_shape[self.axis] = int(x_shape[self.axis] / self.output_num)
        out_shapes = []
        out_dtypes = []
        for _ in range(self.output_num):
            out_shapes.append(tuple(x_shape))
            out_dtypes.append(x['dtype'])
        out_shapes = tuple(out_shapes)
        out_dtypes = tuple(out_dtypes)
        out = {'shape': out_shapes,
               'dtype': out_dtypes,
               'value': None}
        return out


class Rank(PrimitiveWithInfer):
    """
    Returns the rank of a tensor.

    Returns a 0-D int32 Tensor representing the rank of input; the rank of a tensor
    is the number of indices required to uniquely select each element of the tensor.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        Tensor. 0-D int32 Tensor representing the rank of input, i.e., :math:`R`.

    Examples:
        >>> input_tensor = Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32)
        >>> rank = P.Rank()
        >>> output = rank(input_tensor)
        >>> print(output)
        2
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Rank"""

    def __infer__(self, x):
        validator.check_subclass("x", x['dtype'], mstype.tensor, self.name)
        out = {'shape': None,
               'dtype': None,
               'value': len(x['shape'])}
        return out


class TruncatedNormal(PrimitiveWithInfer):
    """
    Returns a tensor of the specified shape filled with truncated normal values.

    The generated values follow a normal distribution.

    Args:
        seed (int): A integer number used to create random seed. Default: 0.
        dtype (:class:`mindspore.dtype`): Data type. Default: mindspore.float32.

    Inputs:
        - **shape** (tuple[int]) - The shape of the output tensor, is a tuple of positive integer.

    Outputs:
        Tensor, the data type of output tensor is the same as attribute `dtype`.

    Examples:
        >>> shape = (1, 2, 3)
        >>> truncated_normal = P.TruncatedNormal()
        >>> output = truncated_normal(shape)
    """

    @prim_attr_register
    def __init__(self, seed=0, dtype=mstype.float32):
        """Initialize TruncatedNormal"""
        validator.check_value_type('seed', seed, [int], self.name)
        validator.check_types_same_and_valid({'dtype': dtype}, mstype.number_type, self.name)

    def __infer__(self, shape):
        shape_value = shape['value']
        validator.check_value_type("shape", shape_value, [tuple], self.name)
        for i, value in enumerate(shape_value):
            validator.check_positive_int(value, f'{i}th value of shape', self.name)
        out = {'shape': shape_value,
               'dtype': mstype.tensor_type(self.dtype),
               'value': None}
        return out


class Size(PrimitiveWithInfer):
    r"""
    Returns the elements count size of a tensor.

    Returns an int scalar representing the elements size of input, the total number of elements in the tensor.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        int, a scalar representing the elements size of `input_x`, tensor is the number of elements
        in a tensor, :math:`size=x_1*x_2*...x_R`.

    Examples:
        >>> input_tensor = Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32)
        >>> size = P.Size()
        >>> output = size(input_tensor)
        >>> print(output)
        4
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Size"""

    def __infer__(self, x):
        size = 1
        validator.check_subclass("x", x['dtype'], mstype.tensor, self.name)
        shp = x['shape']
        if not shp:
            size = 0
        else:
            size = functools.reduce(lambda x, y: x * y, x['shape'])
        out = {'shape': None,
               'dtype': mstype.int32,
               'value': size}
        return out


class Fill(PrimitiveWithInfer):
    """
    Creates a tensor filled with a scalar value.

    Creates a tensor with shape described by the first argument and fills it with values in the second argument.

    Inputs:
        - **type** (mindspore.dtype) - The specified type of output tensor. Only constant value is allowed.
        - **shape** (tuple) - The specified shape of output tensor. Only constant value is allowed.
        - **value** (scalar) - Value to fill the returned tensor. Only constant value is allowed.

    Outputs:
        Tensor, has the same type and shape as input value.

    Examples:
        >>> fill = P.Fill()
        >>> output = fill(mindspore.float32, (2, 2), 1)
        >>> print(output)
        [[1.0, 1.0],
         [1.0, 1.0]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Fill"""

    def __infer__(self, dtype, dims, x):
        validator.check_value_type("shape", dims['value'], [tuple], self.name)
        validator.check_value_type("value", x['value'], [numbers.Number, bool], self.name)
        for i, item in enumerate(dims['value']):
            validator.check_positive_int(item, f'dims[{i}]', self.name)
        valid_dtypes = [mstype.bool_, mstype.int8, mstype.int16, mstype.int32, mstype.int64,
                        mstype.uint8, mstype.uint32, mstype.uint64,
                        mstype.float16, mstype.float32, mstype.float64]
        validator.check_types_same_and_valid({"value": dtype['value']}, valid_dtypes, self.name)
        x_nptype = mstype.dtype_to_nptype(dtype['value'])
        ret = np.full(dims['value'], x['value'], x_nptype)
        out = {
            'value': Tensor(ret),
            'shape': dims['value'],
            'dtype': x['dtype'],
        }
        return out


class Ones(PrimitiveWithInfer):
    """
    Creates a tensor filled with value ones.

    Creates a tensor with shape described by the first argument and
    fills it with value ones in type of the second argument.

    Inputs:
        - **shape** (tuple) - The specified shape of output tensor. Only constant value is allowed.
        - **type** (mindspore.dtype) - The specified type of output tensor. Only constant value is allowed.

    Outputs:
        Tensor, has the same type and shape as input value.

    Examples:
        >>> ones = P.Ones()
        >>> Ones((2, 2), mindspore.float32)
        [[1.0, 1.0],
         [1.0, 1.0]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Fill"""

    def __infer__(self, dims, dtype):
        validator.check_value_type("shape", dims['value'], [tuple], self.name)
        for i, item in enumerate(dims['value']):
            validator.check_positive_int(item, f'dims[{i}]', self.name)
        valid_types = [mstype.bool_, mstype.int8, mstype.int16, mstype.int32, mstype.int64,
                       mstype.uint8, mstype.uint32, mstype.uint64,
                       mstype.float16, mstype.float32, mstype.float64]
        validator.check_types_same_and_valid({"value": dtype['value']}, valid_types, self.name)
        x_nptype = mstype.dtype_to_nptype(dtype['value'])
        ret = np.ones(dims['value'], x_nptype)
        out = {
            'value': Tensor(ret),
            'shape': dims['value'],
            'dtype': x_nptype,
        }
        return out


class Zeros(PrimitiveWithInfer):
    """
    Creates a tensor filled with value zeros.

    Creates a tensor with shape described by the first argument and
    fills it with value zeros in type of the second argument.

    Inputs:
        - **shape** (tuple) - The specified shape of output tensor. Only constant value is allowed.
        - **type** (mindspore.dtype) - The specified type of output tensor. Only constant value is allowed.

    Outputs:
        Tensor, has the same type and shape as input value.

    Examples:
        >>> zeros = P.Zeros()
        >>> Zeros((2, 2), mindspore.float32)
        [[0.0, 0.0],
         [0.0, 0.0]]

    """

    @prim_attr_register
    def __init__(self):
        """Initialize Fill"""

    def __infer__(self, dims, dtype):
        validator.check_value_type("shape", dims['value'], [tuple], self.name)
        for i, item in enumerate(dims['value']):
            validator.check_positive_int(item, f'dims[{i}]', self.name)
        valid_types = [mstype.bool_, mstype.int8, mstype.int16, mstype.int32, mstype.int64,
                       mstype.uint8, mstype.uint32, mstype.uint64,
                       mstype.float16, mstype.float32, mstype.float64]
        validator.check_types_same_and_valid({"value": dtype['value']}, valid_types, self.name)
        x_nptype = mstype.dtype_to_nptype(dtype['value'])
        ret = np.zeros(dims['value'], x_nptype)
        out = {
            'value': Tensor(ret),
            'shape': dims['value'],
            'dtype': x_nptype,
        }
        return out


class OnesLike(PrimitiveWithInfer):
    """
    Creates a new tensor. The values of all elements are 1.

    Returns a tensor of ones with the same shape and type as the input.

    Inputs:
        - **input_x** (Tensor) - Input tensor.

    Outputs:
        Tensor, has the same shape and type as `input_x` but filled with ones.

    Examples:
        >>> oneslike = P.OnesLike()
        >>> x = Tensor(np.array([[0, 1], [2, 1]]).astype(np.int32))
        >>> output = oneslike(x)
        >>> print(output)
        [[1, 1],
         [1, 1]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize OnesLike"""

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('x', x_dtype, mstype.number_type + (mstype.bool_,), self.name)
        return x_dtype


class ZerosLike(PrimitiveWithCheck):
    """
    Creates a new tensor. All elements value are 0.

    Returns a tensor of zeros with the same shape and data type as the input tensor.

    Inputs:
        - **input_x** (Tensor) - Input tensor.

    Outputs:
        Tensor, has the same shape and data type as `input_x` but filled with zeros.

    Examples:
        >>> zeroslike = P.ZerosLike()
        >>> x = Tensor(np.array([[0, 1], [2, 1]]).astype(np.float32))
        >>> output = zeroslike(x)
        >>> print(output)
        [[0.0, 0.0],
         [0.0, 0.0]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize ZerosLike"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def check_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('x', x_dtype, mstype.number_type + (mstype.bool_,), self.name)


class TupleToArray(PrimitiveWithInfer):
    """
    Converts a tuple to a tensor.

    If the type of the first number in the tuple is integer, the data type of the output tensor is int.
    Otherwise, the data type of the output tensor is float.

    Inputs:
        - **input_x** (tuple) - A tuple of numbers. These numbers have the same type. Only constant value is allowed.

    Outputs:
        Tensor, if the input tuple contains `N` numbers, then the shape of the output tensor is (N,).

    Examples:
        >>> type = P.TupleToArray()((1,2,3))
        >>> print(type)
        [1 2 3]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize TupleToArray"""

    def infer_value(self, x):
        validator.check_value_type("x", x, [tuple], self.name)
        validator.check("size of x", len(x), '', 0, Rel.GT, self.name)
        dtype = type(x[0])
        for i, item in enumerate(x):
            validator.check_value_type(f"x[{i}]", item, [numbers.Number], self.name)
        if not all(isinstance(item, dtype) for item in x):
            raise TypeError("For \'{self.name}\' all elements of input x must be have same type.")
        if isinstance(x[0], int):
            ret = np.array(x, np.int32)
        else:
            ret = np.array(x, np.float32)
        return Tensor(ret)

    def __call__(self, x):
        args = list()
        if isinstance(x, range):
            args.append(tuple(x))
        else:
            args.append(x)
        return _run_op(self, self.name, args)


class ScalarToArray(PrimitiveWithInfer):
    """
    Converts a scalar to a `Tensor`.

    Inputs:
        - **input_x** (Union[int, float]) - The input is a scalar. Only constant value is allowed.

    Outputs:
        Tensor. 0-D Tensor and the content is the input.

    Examples:
        >>> op = P.ScalarToArray()
        >>> data = 1.0
        >>> output = op(data)
        >>> print(output)
        1.0
    """

    @prim_attr_register
    def __init__(self):
        pass

    def infer_value(self, x):
        validator.check_value_type("x", x, [int, float], self.name)
        if isinstance(x, int):
            ret = np.array(x, np.int32)
        else:
            ret = np.array(x, np.float32)
        return Tensor(ret)


class ScalarToTensor(PrimitiveWithInfer):
    """
    Converts a scalar to a `Tensor`, and convert data type to specified type.

    Inputs:
        - **input_x** (Union[int, float]) - The input is a scalar. Only constant value is allowed.
        - **dtype** (mindspore.dtype) - The target data type. Default: mindspore.float32. Only
          constant value is allowed.

    Outputs:
        Tensor. 0-D Tensor and the content is the input.

    Examples:
        >>> op = P.ScalarToTensor()
        >>> data = 1
        >>> output = op(data, mindspore.float32)
        >>> print(output)
        1.0
    """

    @prim_attr_register
    def __init__(self):
        pass

    def infer_value(self, x, dtype=mstype.float32):
        validator.check_value_type("x", x, [int, float], self.name)
        validator.check_subclass("dtype", dtype, mstype.number, self.name)
        data_type = mstype.dtype_to_nptype(dtype)
        return Tensor(np.array(x, data_type))


class InvertPermutation(PrimitiveWithInfer):
    r"""
    Computes the inverse of an index permutation.

    Given a tuple input, this operation inserts a dimension of 1 at the dimension
    This operation calculates the inverse of the index replacement. It requires a
    1-dimensional tuple x, which represents the array starting at zero,
    and swaps each value with its index position. In other words, for the output
    tuple y and the input tuple x, this operation calculates the following:
    :math:`y[x[i]] = i, \quad i \in [0, 1, \ldots, \text{len}(x)-1]`.

    Note:
        These values must include 0. There must be no duplicate values and the
        values can not be negative.

    Inputs:
        - **input_x** (Union(tuple[int], list[int]) - The input is constructed by multiple
          integers, i.e., :math:`(y_1, y_2, ..., y_S)` representing the indices.
          The values must include 0. There can be no duplicate values or negative values.
          Only constant value is allowed. The maximum value msut be equal to length of input_x.

    Outputs:
        tuple[int]. It has the same length as the input.

    Examples:
        >>> invert = P.InvertPermutation()
        >>> input_data = (3, 4, 0, 2, 1)
        >>> output = invert(input_data)
        >>> output == (2, 4, 3, 0, 1)
    """

    @prim_attr_register
    def __init__(self):
        """Initialize InvertPermutation"""
        self.set_const_prim(True)

    def __infer__(self, x):
        x_shp = x['shape']
        x_value = x['value']
        if x_value is None:
            raise ValueError(f'For \'{self.name}\' the input value must be const.')
        validator.check_value_type("shape", x_shp, [tuple, list], self.name)
        if mstype.issubclass_(x['dtype'], mstype.tensor):
            raise ValueError(f'For \'{self.name}\' the input value must be non-Tensor.')
        for shp in x_shp:
            if shp != []:
                x_rank = len(np.array(x_value, np.int64).shape)
                raise ValueError(f'For \'{self.name}\' the rank of input must be 1, but got {x_rank}.')
        for i, value in enumerate(x_value):
            validator.check_value_type("input[%d]" % i, value, [int], self.name)
        z = [x_value[i] for i in range(len(x_value))]
        z.sort()

        for i in range(1, len(z)):
            if z[i - 1] == z[i]:
                raise ValueError(f"For {self.name}, {z[i]} is duplicated in the input.")
        validator.check(f'value min', min(x_value), '', 0, Rel.EQ, self.name)
        validator.check(f'value max', max(x_value), '', len(x_value) - 1, Rel.EQ, self.name)

        y = [None] * len(x_value)
        for i, value in enumerate(x_value):
            validator.check_value_type("input[%d]" % i, value, [int], self.name)
            validator.check(f'value', z[i], f'index', i, Rel.EQ, self.name)
            y[value] = i
            z.append(value)
        return {'shape': x_shp,
                'dtype': x['dtype'],
                'value': tuple(y)}


class Argmax(PrimitiveWithInfer):
    """
    Returns the indices of the max value of a tensor across the axis.

    If the shape of input tensor is :math:`(x_1, ..., x_N)`, the shape of the output tensor will be
    :math:`(x_1, ..., x_{axis-1}, x_{axis+1}, ..., x_N)`.

    Args:
        axis (int): Axis where the Argmax operation applies to. Default: -1.
        output_type (:class:`mindspore.dtype`): An optional data type of `mindspore.dtype.int32`.
            Default: `mindspore.dtype.int32`.

    Inputs:
        - **input_x** (Tensor) - Input tensor.

    Outputs:
        Tensor, indices of the max value of input tensor across the axis.

    Examples:
        >>> input_x = Tensor(np.array([2.0, 3.1, 1.2]), mindspore.float32)
        >>> index = P.Argmax(output_type=mindspore.int32)(input_x)
        >>> print(index)
        1
    """

    @prim_attr_register
    def __init__(self, axis=-1, output_type=mstype.int32):
        """Initialize Argmax"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])
        validator.check_value_type("axis", axis, [int], self.name)
        validator.check_types_same_and_valid({'output': output_type}, [mstype.int32], self.name)
        self.axis = axis
        self.add_prim_attr('output_type', output_type)

    def infer_shape(self, x_shape):
        axis = self.axis
        if axis is None:
            axis = 0
        x_rank = len(x_shape)
        validator.check_int_range(axis, -x_rank, x_rank, Rel.INC_LEFT, "axis", self.name)
        axis = axis + x_rank if axis < 0 else axis
        ouput_shape = [x_shape[i] for i in range(x_rank) if i != axis]
        return ouput_shape

    def infer_dtype(self, x_dtype):
        validator.check_subclass("input_x", x_dtype, mstype.tensor, self.name)
        return mstype.tensor_type(self.output_type)


class Argmin(PrimitiveWithInfer):
    """
    Returns the indices of the min value of a tensor across the axis.

    If the shape of input tensor is :math:`(x_1, ..., x_N)`, the shape of the output tensor is
    :math:`(x_1, ..., x_{axis-1}, x_{axis+1}, ..., x_N)`.

    Args:
        axis (int): Axis where the Argmin operation applies to. Default: -1.
        output_type (:class:`mindspore.dtype`): An optional data type of `mindspore.dtype.int32`.
            Default: `mindspore.dtype.int32`.

    Inputs:
        - **input_x** (Tensor) - Input tensor.

    Outputs:
        Tensor, indices of the min value of input tensor across the axis.

    Examples:
        >>> input_x = Tensor(np.array([2.0, 3.1, 1.2]), mindspore.float32)
        >>> index = P.Argmin()(input_x)
        >>> print(index)
        2
    """

    @prim_attr_register
    def __init__(self, axis=-1, output_type=mstype.int32):
        """Initialize Argmin"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])
        validator.check_value_type("axis", axis, [int], self.name)
        validator.check_type_name("output_type", output_type, [mstype.int32, mstype.int64], self.name)
        self.axis = axis
        self.add_prim_attr('output_type', output_type)

    def infer_shape(self, x_shape):
        axis = self.axis
        if axis is None:
            axis = 0
        x_rank = len(x_shape)
        validator.check_int_range(axis, -x_rank, x_rank, Rel.INC_LEFT, "axis", self.name)
        axis = axis + x_rank if axis < 0 else axis
        ouput_shape = [x_shape[i] for i in range(x_rank) if i != axis]
        return ouput_shape

    def infer_dtype(self, x_dtype):
        validator.check_subclass("input_x", x_dtype, mstype.tensor, self.name)
        return mstype.tensor_type(self.output_type)


class ArgMaxWithValue(PrimitiveWithInfer):
    """
    Calculates the maximum value with the corresponding index.

    Calculates the maximum value along with the given axis for the input tensor. It returns the maximum values and
    indices.

    Note:
        In auto_parallel and semi_auto_parallel mode, the first output index can not be used.

    Args:
        axis (int): The dimension to reduce. Default: 0.
        keep_dims (bool): Whether to reduce dimension, if true, the output will keep same dimension with the input,
                          the output will reduce dimension if false. Default: False.

    Inputs:
        - **input_x** (Tensor) - The input tensor, can be any dimension. Set the shape of input tensor as
          :math:`(x_1, x_2, ..., x_N)`.

    Outputs:
        tuple (Tensor), tuple of 2 tensors, containing the corresponding index and the maximum value of the input
        tensor.
        - index (Tensor) - The index for the maximum value of the input tensor. If `keep_dims` is true, the shape of
        output tensors is :math:`(x_1, x_2, ..., x_{axis-1}, 1, x_{axis+1}, ..., x_N)`. Otherwise, the shape is
        :math:`(x_1, x_2, ..., x_{axis-1}, x_{axis+1}, ..., x_N)`.
        - output_x (Tensor) - The maximum value of input tensor, with the same shape as index.

    Examples:
        >>> input_x = Tensor(np.random.rand(5), mindspore.float32)
        >>> index, output = P.ArgMaxWithValue()(input_x)
    """

    @prim_attr_register
    def __init__(self, axis=0, keep_dims=False):
        """Initialize ArgMaxWithValue"""
        self.axis = axis
        self.keep_dims = keep_dims
        validator.check_value_type('keep_dims', keep_dims, [bool], self.name)
        validator.check_value_type('axis', axis, [int], self.name)

    def infer_shape(self, x_shape):
        axis = self.axis
        x_rank = len(x_shape)
        validator.check_int_range(axis, -x_rank, x_rank, Rel.INC_LEFT, "axis", self.name)
        ouput_shape = _infer_shape_reduce(x_shape, self.axis, self.keep_dims, self.name)
        return ouput_shape, ouput_shape

    def infer_dtype(self, x_dtype):
        validator.check_subclass("input_x", x_dtype, mstype.tensor, self.name)
        return mstype.tensor_type(mstype.int32), x_dtype


class ArgMinWithValue(PrimitiveWithInfer):
    """
    Calculates the minimum value with corresponding index, return indices and values.

    Calculates the minimum value along with the given axis for the input tensor. It returns the minimum values and
    indices.

    Note:
        In auto_parallel and semi_auto_parallel mode, the first output index can not be used.

    Args:
        axis (int): The dimension to reduce. Default: 0.
        keep_dims (bool): Whether to reduce dimension, if true the output will keep the same dimension as the input,
                          the output will reduce dimension if false. Default: False.

    Inputs:
        - **input_x** (Tensor) - The input tensor, can be any dimension. Set the shape of input tensor as
          :math:`(x_1, x_2, ..., x_N)`.

    Outputs:
        tuple (Tensor), tuple of 2 tensors, containing the corresponding index and the minimum value of the input
        tensor.
        - index (Tensor) - The index for the minimum value of the input tensor. If `keep_dims` is true, the shape of
        output tensors is :math:`(x_1, x_2, ..., x_{axis-1}, 1, x_{axis+1}, ..., x_N)`. Otherwise, the shape is
        :math:`(x_1, x_2, ..., x_{axis-1}, x_{axis+1}, ..., x_N)`.
        - output_x (Tensor) - The minimum value of input tensor, with the same shape as index.

    Examples:
        >>> input_x = Tensor(np.random.rand(5), mindspore.float32)
        >>> index, output = P.ArgMinWithValue()(input_x)
        >>> print((index, output))
        0 0.0496291
    """

    @prim_attr_register
    def __init__(self, axis=0, keep_dims=False):
        """Initialize ArgMinWithValue"""
        self.axis = axis
        self.keep_dims = keep_dims
        validator.check_value_type('keep_dims', keep_dims, [bool], self.name)
        validator.check_value_type('axis', axis, [int], self.name)

    def infer_shape(self, x_shape):
        axis = self.axis
        x_rank = len(x_shape)
        validator.check_int_range(axis, -x_rank, x_rank, Rel.INC_LEFT, "axis", self.name)
        ouput_shape = _infer_shape_reduce(x_shape, self.axis, self.keep_dims, self.name)
        return ouput_shape, ouput_shape

    def infer_dtype(self, x_dtype):
        validator.check_subclass("input_x", x_dtype, mstype.tensor, self.name)
        return mstype.tensor_type(mstype.int32), x_dtype


class Tile(PrimitiveWithInfer):
    r"""
    Replicates a tensor with given multiples times.

    Creates a new tensor by replicating input multiples times. The dimension of
    output tensor is the larger of the input tensor dimension and the length of `multiples`.

    Inputs:
        - **input_x** (Tensor) - 1-D or higher Tensor. Set the shape of input tensor as
          :math:`(x_1, x_2, ..., x_S)`.

        - **multiples** (tuple[int]) - The input tuple is constructed by multiple
          integers, i.e., :math:`(y_1, y_2, ..., y_S)`. The length of `multiples`
          cannot be smaller than the length of the shape of `input_x`.
          Only constant value is allowed.

    Outputs:
        Tensor, has the same data type as the `input_x`.

        - If the length of `multiples` is the same as the length of shape of `input_x`,
          then the shape of their corresponding positions can be multiplied, and
          the shape of Outputs is :math:`(x_1*y_1, x_2*y_2, ..., x_S*y_R)`.
        - If the length of `multiples` is larger than the length of shape of `input_x`,
          fill in multiple 1 in the length of the shape of `input_x` until their lengths are consistent.
          Such as set the shape of `input_x` as :math:`(1, ..., x_1, x_2, ..., x_S)`,
          then the shape of their corresponding positions can be multiplied, and
          the shape of Outputs is :math:`(1*y_1, ..., x_S*y_R)`.

    Examples:
        >>> tile = P.Tile()
        >>> input_x = Tensor(np.array([[1, 2], [3, 4]]), mindspore.float32)
        >>> multiples = (2, 3)
        >>> result = tile(input_x, multiples)
        >>> print(result)
        [[1.  2.  1.  2.  1.  2.]
         [3.  4.  3.  4.  3.  4.]
         [1.  2.  1.  2.  1.  2.]
         [3.  4.  3.  4.  3.  4.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Tile"""
        self.init_prim_io_names(inputs=['x', 'multiples'], outputs=['output'])

    def check_elim(self, base_tensor, multiplier):
        if (not isinstance(base_tensor, Tensor)) or (not isinstance(multiplier, tuple)):
            raise TypeError("Expecting (Tensor, tuple), got: ({}, {})".format(base_tensor, multiplier))
        if all(v == 1 for v in multiplier):
            return (True, base_tensor)
        return (False, None)

    def __infer__(self, x, multiples):
        multiples_v = multiples['value']
        x_shp = x['shape']
        validator.check_value_type("multiples", multiples_v, [tuple], self.name)
        for i, multiple in enumerate(multiples_v):
            validator.check_positive_int(multiple, "multiples[%d]" % i, self.name)
        validator.check_value_type("x[\'dtype\']", x["dtype"], mstype.tensor_type, self.name)
        len_sub = len(multiples_v) - len(x_shp)
        multiples_w = None
        if len_sub == 0:
            multiples_w = multiples_v
        if len_sub > 0:
            for i in range(0, len_sub):
                x_shp.insert(0, 1)
            multiples_w = multiples_v
        elif len_sub < 0:
            raise ValueError(f'For \'{self.name}\' the length of multiples can not be smaller than '
                             f'the length of dimension in input_x.')
        for i, a in enumerate(multiples_w):
            x_shp[i] *= a
        value = None
        if x['value'] is not None:
            value = Tensor(np.tile(x['value'].asnumpy(), multiples_w))
        return {'shape': x_shp,
                'dtype': x['dtype'],
                'value': value}


class UnsortedSegmentSum(PrimitiveWithInfer):
    r"""
    Computes the sum along segments of a tensor.

    Calculates a tensor such that :math:`\text{output}[i] = \sum_{segment\_ids[j] == i} \text{data}[j, \ldots]`, where
    :math:`j` is a tuple describing the index of element in data.  `segment_ids` selects which elements in data to sum
    up. Segment_ids does not need to be sorted, and it does not need to cover all values in the entire valid value
    range.

    If the sum of the given segment_ids :math:`i` is empty, then :math:`\text{output}[i] = 0`. If the given segment_ids
    is negative, the value will be ignored. 'num_segments' must be equal to the number of different segment_ids.

    Inputs:
        - **input_x** (Tensor) - The shape is :math:`(x_1, x_2, ..., x_R)`.
        - **segment_ids** (Tensor) - Set the shape as :math:`(x_1, x_2, ..., x_N)`, where 0 < N <= R. Type must be int.
        - **num_segments** (int) - Set :math:`z` as num_segments.

    Outputs:
        Tensor, the shape is :math:`(z, x_{N+1}, ..., x_R)`.

    Examples:
        >>> input_x = Tensor([1, 2, 3, 4], mindspore.float32)
        >>> segment_ids = Tensor([0, 0, 1, 2], mindspore.int32)
        >>> num_segments = 4
        >>> output = P.UnsortedSegmentSum()(input_x, segment_ids, num_segments)
        >>> print(output)
        [3, 3, 4, 0]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize UnsortedSegmentSum"""
        self.init_prim_io_names(inputs=['x', 'segment_ids', 'num_segments'], outputs=['y'])
        self.add_prim_attr("dynamic_shape_depends", [2])

    def __infer__(self, x, segment_ids, num_segments):
        x_type = x['dtype']
        x_shp = x['shape']
        validator.check_subclass("input_x", x_type, mstype.tensor, self.name)
        validator.check_value_type("x_shape", x_shp, [list], self.name)
        x_shp_len = len(x_shp)
        validator.check_positive_int(x_shp_len, "rank of input_x", self.name)
        segment_ids_shp = segment_ids['shape']
        segment_ids_type = segment_ids['dtype']
        validator.check_subclass("segment_ids", segment_ids_type, mstype.tensor, self.name)
        validator.check_value_type("segment_ids", segment_ids_shp, [list], self.name)
        segment_ids_shp_len = len(segment_ids_shp)
        validator.check_positive_int(segment_ids_shp_len, "rank of segment_ids", self.name)
        validator.check(f'rank of input_x', len(x_shp),
                        'rank of segments_id', len(segment_ids_shp), Rel.GE, self.name)
        for i, value in enumerate(segment_ids_shp):
            validator.check("ids[%d]" % i, value, 'input[%d]' % i, x_shp[i], Rel.EQ, self.name)
        num_segments_v = num_segments['value']
        num_segments_type = num_segments['dtype']
        validator.check_subclass("num_segments", num_segments_type, [mstype.tensor, mstype.number], self.name)
        if isinstance(num_segments_type, type(mstype.tensor)):
            validator.check_tensor_dtype_valid("num_segments", num_segments_type, [mstype.int32, mstype.int64],
                                               self.name)
            shp = [-1]
        else:
            validator.check_value_type('num_segments', num_segments_v, [int], self.name)
            validator.check_positive_int(num_segments_v, "num_segments", self.name)
            shp = [num_segments_v]

        shp += x_shp[segment_ids_shp_len:]
        if 'max_shape' in x:
            output_max_shape = x['max_shape']
        else:
            output_max_shape = x_shp
        out = {'shape': shp,
               'max_shape': output_max_shape,
               'min_shape': [1] * segment_ids_shp_len + x_shp[segment_ids_shp_len:],
               'dtype': mstype.tensor_type(x_type.element_type()),
               'value': None}
        return out


class UnsortedSegmentMin(PrimitiveWithInfer):
    """
    Computes the minimum along segments of a tensor.

    Inputs:
        - **input_x** (Tensor) - The shape is :math:`(x_1, x_2, ..., x_R)`.
          The data type must be float16, float32 or int32.
        - **segment_ids** (Tensor) - A `1-D` tensor whose shape is :math:`(x_1)`, the value must be >= 0.
          The data type must be int32.
        - **num_segments** (int) - The value spcifies the number of distinct `segment_ids`.

    Outputs:
        Tensor, set the number of `num_segments` as `N`, the shape is :math:`(N, x_2, ..., x_R)`.

    Examples:
        >>> input_x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [4, 2, 1]]).astype(np.float32))
        >>> segment_ids = Tensor(np.array([0, 1, 1]).astype(np.int32))
        >>> num_segments = 2
        >>> unsorted_segment_min = P.UnsortedSegmentMin()
        >>> unsorted_segment_min(input_x, segment_ids, num_segments)
        [[1., 2., 3.], [4., 2., 1.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize UnsortedSegmentMin"""
        self.init_prim_io_names(inputs=['x', 'segment_ids', 'num_segments'], outputs=['y'])

    def __infer__(self, x, segment_ids, num_segments):
        x_type = x['dtype']
        x_shape = x['shape']
        segment_ids_shape = segment_ids['shape']
        valid_type = [mstype.float16, mstype.float32, mstype.int32]
        validator.check_tensor_dtype_valid("x", x['dtype'], valid_type, self.name)
        validator.check_tensor_dtype_valid("segment_ids", segment_ids['dtype'], [mstype.int32], self.name)
        validator.check_equal_int(len(segment_ids_shape), 1, "rank of segment_ids_shape", self.name)
        validator.check(f'first shape of input_x', x_shape[0],
                        'length of segments_id', segment_ids_shape[0], Rel.EQ, self.name)
        num_segments_v = num_segments['value']
        validator.check_value_type('num_segments', num_segments_v, [int], self.name)
        validator.check_positive_int(num_segments_v, "num_segments", self.name)
        segment_ids_shape_len = len(segment_ids_shape)
        out_shape = [num_segments_v]
        out_shape += x_shape[segment_ids_shape_len:]
        out = {'shape': out_shape,
               'dtype': x_type,
               'value': None}
        return out


class UnsortedSegmentMax(PrimitiveWithInfer):
    """
    Computes the maximum along segments of a tensor.

    Inputs:
        - **input_x** (Tensor) - The shape is :math:`(x_1, x_2, ..., x_R)`.
          The data type must be float16, float32 or int32.
        - **segment_ids** (Tensor) - A `1-D` tensor whose shape is :math:`(x_1)`, the value must be >= 0.
          The data type must be int32.
        - **num_segments** (int) - The value spcifies the number of distinct `segment_ids`.

    Outputs:
        Tensor, set the number of `num_segments` as `N`, the shape is :math:`(N, x_2, ..., x_R)`.

    Examples:
        >>> input_x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [4, 2, 1]]).astype(np.float32))
        >>> segment_ids = Tensor(np.array([0, 1, 1]).astype(np.int32))
        >>> num_segments = 2
        >>> unsorted_segment_max = P.UnsortedSegmentMax()
        >>> unsorted_segment_max(input_x, segment_ids, num_segments)
        [[1., 2., 3.], [4., 5., 6.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize UnsortedSegmentMax"""
        self.init_prim_io_names(inputs=['x', 'segment_ids', 'num_segments'], outputs=['y'])

    def __infer__(self, x, segment_ids, num_segments):
        x_type = x['dtype']
        x_shape = x['shape']
        segment_ids_shape = segment_ids['shape']
        valid_type = [mstype.float16, mstype.float32, mstype.int32]
        validator.check_tensor_dtype_valid("x", x['dtype'], valid_type, self.name)
        validator.check_tensors_dtypes_same_and_valid({"segment_ids": segment_ids['dtype']}, [mstype.int32], self.name)
        validator.check_equal_int(len(segment_ids_shape), 1, "rank of segment_ids_shape", self.name)
        validator.check(f'first shape of input_x', x_shape[0],
                        'length of segments_id', segment_ids_shape[0], Rel.EQ, self.name)
        num_segments_v = num_segments['value']
        validator.check_value_type('num_segments', num_segments_v, [int], self.name)
        validator.check_positive_int(num_segments_v, "num_segments", self.name)
        segment_ids_shape_len = len(segment_ids_shape)
        out_shape = [num_segments_v]
        out_shape += x_shape[segment_ids_shape_len:]
        out = {'shape': out_shape,
               'dtype': x_type,
               'value': None}
        return out


class UnsortedSegmentProd(PrimitiveWithInfer):
    """
    Computes the product along segments of a tensor.

    Inputs:
        - **input_x** (Tensor) - The shape is :math:`(x_1, x_2, ..., x_R)`.
          With float16, float32 or int32 data type.
        - **segment_ids** (Tensor) - A `1-D` tensor whose shape is :math:`(x_1)`, the value must be >= 0.
          Data type must be int32.
        - **num_segments** (int) - The value spcifies the number of distinct `segment_ids`,
          must be greater than 0.

    Outputs:
        Tensor, set the number of `num_segments` as `N`, the shape is :math:`(N, x_2, ..., x_R)`.

    Examples:
        >>> input_x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [4, 2, 1]]).astype(np.float32))
        >>> segment_ids = Tensor(np.array([0, 1, 0]).astype(np.int32))
        >>> num_segments = 2
        >>> unsorted_segment_prod = P.UnsortedSegmentProd()
        >>> unsorted_segment_prod(input_x, segment_ids, num_segments)
        [[4., 4., 3.], [4., 5., 6.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize UnsortedSegmentProd"""
        self.init_prim_io_names(inputs=['x', 'segment_ids', 'num_segments'], outputs=['y'])

    def __infer__(self, x, segment_ids, num_segments):
        x_type = x['dtype']
        x_shape = x['shape']
        segment_ids_shape = segment_ids['shape']
        validator.check_subclass("input_x", x_type, mstype.tensor, self.name)
        validator.check_value_type("x_shape", x_shape, [list], self.name)
        valid_type = [mstype.float16, mstype.float32, mstype.int32]
        validator.check_tensor_dtype_valid("x", x['dtype'], valid_type, self.name)
        validator.check_tensor_dtype_valid("segment_ids", segment_ids['dtype'], [mstype.int32], self.name)
        validator.check_equal_int(len(segment_ids_shape), 1, "rank of segment_ids_shape", self.name)
        validator.check(f'first shape of input_x', x_shape[0],
                        'length of segments_id', segment_ids_shape[0], Rel.EQ, self.name)
        num_segments_v = num_segments['value']
        validator.check_value_type('num_segments', num_segments_v, [int], self.name)
        validator.check_positive_int(num_segments_v, "num_segments", self.name)
        segment_ids_shape_len = len(segment_ids_shape)
        out_shape = [num_segments_v]
        out_shape += x_shape[segment_ids_shape_len:]
        out = {'shape': out_shape,
               'dtype': mstype.tensor_type(x_type.element_type()),
               'value': None}
        return out


class Concat(PrimitiveWithInfer):
    r"""
    Concats tensor in specified axis.

    Concats input tensors along with the given axis.

    Note:
        The input data is a tuple of tensors. These tensors have the same rank `R`. Set the given axis as `m`, and
        :math:`0 \le m < R`. Set the number of input tensors as `N`. For the :math:`i`-th tensor :math:`t_i`, it has
        the shape of :math:`(x_1, x_2, ..., x_{mi}, ..., x_R)`. :math:`x_{mi}` is the :math:`m`-th dimension of the
        :math:`i`-th tensor. Then, the shape of the output tensor is

        .. math::
            (x_1, x_2, ..., \sum_{i=1}^Nx_{mi}, ..., x_R)

    Args:
        axis (int): The specified axis. Default: 0.

    Inputs:
        - **input_x** (tuple, list) - A tuple or a list of input tensors.

    Outputs:
        Tensor, the shape is :math:`(x_1, x_2, ..., \sum_{i=1}^Nx_{mi}, ..., x_R)`.

    Examples:
        >>> data1 = Tensor(np.array([[0, 1], [2, 1]]).astype(np.int32))
        >>> data2 = Tensor(np.array([[0, 1], [2, 1]]).astype(np.int32))
        >>> op = P.Concat()
        >>> output = op((data1, data2))
        >>> print(output)
        [[0, 1],
         [2, 1],
         [0, 1],
         [2, 1]]
    """

    @prim_attr_register
    def __init__(self, axis=0):
        """Initialize Tile"""
        validator.check_value_type("axis", axis, [int], self.name)

    def __infer__(self, input_x):
        axis = self.axis
        x_shp = input_x['shape']
        x_type = input_x['dtype']
        _, all_shp, _ = get_concat_offset(x_shp, x_type, axis, self.name)
        self.add_prim_attr('T', x_type[0].element_type())
        self.add_prim_attr('inputNums', len(x_shp))
        ret_shp = x_shp[0].copy()
        ret_shp[axis] = all_shp
        out = {'shape': ret_shp,
               'dtype': x_type[0],
               'value': None}
        return out


class ParallelConcat(PrimitiveWithInfer):
    r"""
    Concats tensor in the first dimension.

    Concats input tensors along with the first dimension.

    Note:
        The input tensors are all required to have size 1 in the first dimension.

    Inputs:
        - **values** (tuple, list) - A tuple or a list of input tensors. The data type and shape of these
          tensors must be the same.

    Outputs:
        Tensor, data type is the same as `values`.

    Examples:
        >>> data1 = Tensor(np.array([[0, 1]]).astype(np.int32))
        >>> data2 = Tensor(np.array([[2, 1]]).astype(np.int32))
        >>> op = P.ParallelConcat()
        >>> output = op((data1, data2))
        >>> print(output)
        [[0, 1], [2, 1]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize ParallelConcat"""

    def __infer__(self, values):
        x_shp = values['shape']
        x_type = values['dtype']

        validator.check_int(len(x_shp), 1, Rel.GE, f'x_shp length', self.name)

        args = {f"x_type[{i}]": elem for i, elem in enumerate(x_type)}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.number_type + (mstype.bool_,), self.name)

        first_elem = x_shp[0]
        for i, elem in enumerate(x_shp[1:]):
            j = i + 1
            validator.check_equal_int(elem[0], 1, f'x_shp[{j}][0]', self.name)
            validator.check(f"x_shp[0] shape", first_elem, f"x_shp[{j}] shape", elem, Rel.EQ, self.name)

        ret_shp = x_shp[0].copy()
        ret_shp[0] = len(x_shp)
        self.add_prim_attr('shape', ret_shp)
        self.add_prim_attr('N', len(x_shp))

        out = {'shape': ret_shp,
               'dtype': x_type[0],
               'value': None}
        return out


def _get_pack_shape(x_shape, x_type, axis, prim_name):
    """for pack output shape"""
    validator.check_value_type("shape", x_shape, [tuple, list], prim_name)
    validator.check_int(len(x_shape), 1, Rel.GE, "len of input_x", prim_name)
    validator.check_subclass("input_x[0]", x_type[0], mstype.tensor, prim_name)
    rank_base = len(x_shape[0])
    N = len(x_shape)
    out_shape = x_shape[0]
    validator.check_int_range(axis, -rank_base - 1, rank_base, Rel.INC_BOTH, 'axis', prim_name)
    if axis < 0:
        axis = axis + rank_base + 1
    for i in range(1, N):
        validator.check('x_type[%d]' % i, x_type[i], 'base', x_type[0], Rel.EQ, prim_name, TypeError)
        if x_shape[i] != x_shape[0]:
            raise ValueError(f"For \'{prim_name}\' element {i} shape in input can not pack with first element")
    out_shape.insert(axis, N)
    return out_shape


class Pack(PrimitiveWithInfer):
    r"""
    Packs a list of tensors in specified axis.

    Packs the list of input tensors with the same rank `R`, output is a tensor of rank `(R+1)`.

    Given input tensors of shape :math:`(x_1, x_2, ..., x_R)`. Set the number of input tensors as `N`.
    If :math:`0 \le axis`, the shape of the output tensor is :math:`(x_1, x_2, ..., x_{axis}, N, x_{axis+1}, ..., x_R)`.

    Args:
        axis (int): Dimension to pack. Default: 0.
                    Negative values wrap around. The range is [-(R+1), R+1).

    Inputs:
        - **input_x** (Union[tuple, list]) - A Tuple or list of Tensor objects with the same shape and type.

    Outputs:
        Tensor. A packed Tensor with the same type as `input_x`.

    Raises:
        TypeError: If the data types of elements in `input_x` are not the same.
        ValueError: If the length of `input_x` is not greater than 1;
                    or if axis is out of the range [-(R+1), R+1);
                    or if the shapes of elements in input_x are not the same.

    Examples:
        >>> data1 = Tensor(np.array([0, 1]).astype(np.float32))
        >>> data2 = Tensor(np.array([2, 3]).astype(np.float32))
        >>> pack = P.Pack()
        >>> output = pack([data1, data2])
        >>> print(output)
        [[0, 1], [2, 3]]
    """

    @prim_attr_register
    def __init__(self, axis=0):
        """Initialize Pack"""
        validator.check_value_type("axis", axis, [int], self.name)
        self.axis = axis

    def __infer__(self, value):
        x_shape = value['shape']
        x_type = value['dtype']
        self.add_prim_attr('num', len(x_shape))
        all_shape = _get_pack_shape(x_shape, x_type, self.axis, self.name)
        out = {'shape': all_shape,
               'dtype': x_type[0],
               'value': None}
        return out


class Unpack(PrimitiveWithInfer):
    r"""
    Unpacks tensor in specified axis.

    Unpacks a tensor of rank `R` along axis dimension, output tensors will have rank `(R-1)`.

    Given a tensor of shape :math:`(x_1, x_2, ..., x_R)`. If :math:`0 \le axis`,
    the shape of tensor in output is :math:`(x_1, x_2, ..., x_{axis}, x_{axis+2}, ..., x_R)`.

    This is the opposite of pack.

    Args:
        axis (int): Dimension along which to pack. Default: 0.
                    Negative values wrap around. The range is [-R, R).

    Inputs:
        - **input_x** (Tensor) - The shape is :math:`(x_1, x_2, ..., x_R)`.
          A tensor to be unpacked and the rank of the tensor must be greater than 0.

    Outputs:
        A tuple of tensors, the shape of each objects is the same.

    Raises:
        ValueError: If axis is out of the range [-len(input_x.shape), len(input_x.shape)).

    Examples:
        >>> unpack = P.Unpack()
        >>> input_x = Tensor(np.array([[1, 1, 1, 1], [2, 2, 2, 2]]))
        >>> output = unpack(input_x)
        >>> print(output)
        ([1, 1, 1, 1], [2, 2, 2, 2])
    """

    @prim_attr_register
    def __init__(self, axis=0):
        """Initialize Unpack"""
        validator.check_value_type("axis", axis, [int], self.name)
        self.axis = axis

    def __infer__(self, x):
        validator.check_subclass("x", x['dtype'], mstype.tensor, self.name)
        x_shape = list(x['shape'])
        dim = len(x_shape)
        validator.check_int_range(self.axis, -dim, dim, Rel.INC_LEFT, 'axis value', self.name)
        if self.axis < 0:
            self.axis = self.axis + dim
        output_num = x_shape[self.axis]
        validator.check_value_type("num", output_num, [int], self.name)
        validator.check_positive_int(output_num, "output_num", self.name)
        self.add_prim_attr('num', output_num)
        output_valid_check = x_shape[self.axis] - output_num
        validator.check_int(output_valid_check, 0, Rel.EQ,
                            "The dimension which to unpack divides output_num", self.name)
        out_shapes = []
        out_dtypes = []
        out_shape = x_shape[:self.axis] + x_shape[self.axis + 1:]
        for _ in range(output_num):
            out_shapes.append(tuple(out_shape))
            out_dtypes.append(x['dtype'])
        out_shapes = tuple(out_shapes)
        out_dtypes = tuple(out_dtypes)
        out = {'shape': out_shapes,
               'dtype': out_dtypes,
               'value': None}
        return out


class Slice(PrimitiveWithInfer):
    """
    Slices a tensor in the specified shape.

    Inputs:
        - **x** (Tensor): The target tensor.
        - **begin** (tuple): The beginning of the slice. Only constant value is allowed.
        - **size** (tuple): The size of the slice. Only constant value is allowed.

    Outputs:
        Tensor, the shape is : input `size`, the data type is the same as input `x`.

    Examples:
        >>> data = Tensor(np.array([[[1, 1, 1], [2, 2, 2]],
        ...                         [[3, 3, 3], [4, 4, 4]],
        ...                         [[5, 5, 5], [6, 6, 6]]]).astype(np.int32))
        >>> type = P.Slice()(data, (1, 0, 0), (1, 1, 3))
        >>> print(type)
        [[[3 3 3]]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize slice"""
        self.init_prim_io_names(inputs=['x', 'begin', 'size'], outputs=['output'])

    def __infer__(self, x, begin, size):
        x_shape = x['shape']
        x_shp_len = len(x_shape)
        validator.check_const_input('begin', begin['value'], self.name)
        validator.check_const_input('size', size['value'], self.name)
        begin_v, size_v = begin['value'], size['value']
        if begin_v is None or size_v is None:
            return {'shape': None,
                    'dtype': x['dtype'],
                    'value': None}
        for key, value in zip(('begin', 'size'), (begin_v, size_v)):
            validator.check(f'len of {key}', len(value),
                            'len x\'s dim', x_shp_len)
        for i in range(x_shp_len):
            if x_shape[i] < begin_v[i] + size_v[i]:
                y = begin_v[i] + size_v[i]
                raise ValueError("For '%s' slice shape can not bigger than orign shape %d, %d." %
                                 (self.name, x_shape[i], y))
        return {'shape': size_v,
                'dtype': x['dtype'],
                'value': None}


class ReverseV2(PrimitiveWithInfer):
    """
    Reverses specific dimensions of a tensor.

    Args:
        axis (Union[tuple(int), list(int)): The indices of the dimensions to reverse.

    Inputs:
        - **input_x** (Tensor) - The target tensor.

    Outputs:
        Tensor, has the same shape and type as `input_x`.

    Examples:
        >>> input_x = Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), mindspore.int32)
        >>> op = P.ReverseV2(axis=[1])
        >>> output = op(input_x)
        >>> print(output)
        [[4, 3, 2, 1], [8, 7, 6, 5]]
    """

    @prim_attr_register
    def __init__(self, axis):
        validator.check_value_type('axis', axis, [list, tuple], self.name)
        for i, each in enumerate(axis):
            validator.check_value_type(f'axis[{i}]', each, [int], self.name)
        self.axis = axis
        self.init_prim_io_names(inputs=['x'], outputs=['output'])

    def infer_shape(self, x_shape):
        dim = len(x_shape)
        for i, each in enumerate(self.axis):
            validator.check_int_range(each, -dim, dim, Rel.INC_LEFT, f'axis[{i}]', self.name)
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('x', x_dtype, (mstype.bool_,) + mstype.number_type, self.name)
        return x_dtype


class Rint(PrimitiveWithInfer):
    """
    Returns element-wise integer closest to x.

    Inputs:
        - **input_x** (Tensor) - The target tensor, which must be one of the following types:
          float16, float32.

    Outputs:
        Tensor, has the same shape and type as `input_x`.

    Examples:
        >>> input_x = Tensor(np.array([-1.6, -0.1, 1.5, 2.0]), mindspore.float32)
        >>> op = P.Rint()
        >>> output = op(input_x)
        >>> print(output)
        [-2., 0., 2., 2.]
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x'], outputs=['output'])

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('x', x_dtype, [mstype.float16, mstype.float32], self.name)
        return x_dtype


class Select(PrimitiveWithInfer):
    r"""

    Returns the selected elements, either from input :math:`x` or input :math:`y`, depending on the `condition`.

    Given a tensor as input, this operation inserts a dimension of 1 at the dimension,
    if both :math:`x` and :math:`y` are none, the operation returns the coordinates of the true
    element in the `condition`, the coordinates are returned as a two-dimensional
    tensor, where the first dimension (row) represents the number of true elements
    and the second dimension (columns) represents the coordinates of the true
    elements. Keep in mind that the shape of the output tensor can vary depending
    on how many true values are in the input. Indexes are output in row-first
    order.

    If neither is None, :math:`x` and :math:`y` must have the same shape. If :math:`x` and :math:`y` are
    scalars, the conditional tensor must be a scalar. If :math:`x` and :math:`y` are
    higher-demensional vectors, the `condition` must be a vector whose size matches the
    first dimension of :math:`x`, or must have the same shape as :math:`y`.

    The conditional tensor acts as an optional compensation (mask), which
    determines whether the corresponding element / row in the output must be
    selected from :math:`x` (if true) or :math:`y` (if false) based on the value of each
    element.

    If condition is a vector, then :math:`x` and :math:`y` are higher-demensional matrices, then it
    chooses to copy that row (external dimensions) from :math:`x` and :math:`y`. If condition has
    the same shape as :math:`x` and :math:`y`, you can choose to copy these elements from :math:`x`
    and :math:`y`.

    Inputs:
        - **input_cond** (Tensor[bool]) - The shape is :math:`(x_1, x_2, ..., x_N, ..., x_R)`.
          The condition tensor, decides which element is chosen.
        - **input_x** (Tensor) - The shape is :math:`(x_1, x_2, ..., x_N, ..., x_R)`.
          The first input tensor.
        - **input_y** (Tensor) - The shape is :math:`(x_1, x_2, ..., x_N, ..., x_R)`.
          The second input tensor.

    Outputs:
        Tensor, has the same shape as `input_x`. The shape is :math:`(x_1, x_2, ..., x_N, ..., x_R)`.

    Examples:
        >>> select = P.Select()
        >>> input_cond = Tensor([True, False])
        >>> input_x = Tensor([2,3], mindspore.float32)
        >>> input_y = Tensor([1,2], mindspore.float32)
        >>> select(input_cond, input_x, input_y)
        [2. 2.]
    """

    @prim_attr_register
    def __init__(self):
        """init"""
        self.init_prim_io_names(inputs=['condition', 'x', 'y'], outputs=['output'])

    def infer_shape(self, cond_shape, x_shape, y_shape):
        if cond_shape != x_shape or x_shape != y_shape:
            raise ValueError('The x_shape and y_shape must be the same as cond_shape.')
        return x_shape

    def infer_dtype(self, cond_type, x_type, y_type):
        self.add_prim_attr('T', x_type)
        validator.check_subclass("x_type", x_type, mstype.tensor, self.name)
        validator.check_subclass("y_type", y_type, mstype.tensor, self.name)
        validator.check_tensor_dtype_valid("cond", cond_type, [mstype.bool_], self.name)
        if x_type != y_type:
            raise TypeError('\'%s\' the x_type %s must be the same as y_type %s.' % (self.name, x_type, y_type))
        return x_type

    def infer_value(self, cond, x, y):
        if cond is not None and x is not None and y is not None:
            cond = cond.asnumpy()
            x = x.asnumpy()
            y = y.asnumpy()
            out = np.where(cond, x, y)
            return Tensor(out)
        return None


def _compute_slicing_length(begin, end, stride, x_shape, i):
    """Computes the length of the slicing."""
    if i >= len(x_shape):
        raise ValueError(f"For 'StridedSlice', When their is no new axis, the index length must be less or "
                         f"equal than the dim of x.")
    x_dim = x_shape[i]
    if stride > 0:
        # When slicing forward, convert begin and end to positive numbers.
        if begin >= x_dim or end < -x_dim:
            # When slicing forward, if begin >= x_dim or end < -x_dim, the length of the slicing is 0.
            slicing_length = 0
        else:
            if -x_dim <= begin < 0:
                begin += x_dim
            if begin < -x_dim:
                # When slicing forward, if begin < -x_dim, set begin = 0, which means start from the 0th element.
                begin = 0
            if -x_dim <= end < 0:
                end += x_dim
            if end > x_dim:
                # When slicing forward, if end > x_dim, set end = x_dims, which means slice to the last element.
                end = x_dim
            if begin >= end:
                # When slicing forward, if begin >= end, the length of the slicing is 0.
                slicing_length = 0
            else:
                slicing_length = 1 + (end - 1 - begin) // stride
    else:
        # When slicing backward, convert begin and end to negative numbers.
        if begin < -x_dim or end >= x_dim:
            # When slicing backward, if begin < -x_dim or end >= x_dim, the length of the slicing is 0.
            slicing_length = 0
        else:
            if 0 <= begin < x_dim:
                begin += -x_dim
            if begin >= x_dim:
                begin = -1
            if 0 <= end < x_dim:
                end += -x_dim
            if end < -x_dim - 1:
                # When slicing backward, if end < -x_dim - 1, set end = -x_dim - 1, which means
                # slicing to the 0th element.
                end = -x_dim - 1
            if begin <= end:
                # When slicing backward, if begin <= end, the length of the slicing is 0.
                slicing_length = 0
            else:
                slicing_length = 1 + (end + 1 - begin) // stride
    return slicing_length


class StridedSlice(PrimitiveWithInfer):
    r"""

    Extracts a strided slice of a tensor.

    Given an input tensor, this operation inserts a dimension of length 1 at the dimension.
    This operation extracts a fragment of size (end-begin)/stride from the given 'input_tensor'.
    Starting from the begining position, the fragment continues adding stride to the index until
    all dimensions are not less than the ending position.

    Note:
        The stride may be negative value, which causes reverse slicing.
        The shape of `begin`, `end` and `strides` must be the same.

    Args:
        begin_mask (int): Starting index of the slice. Default: 0.
        end_mask (int): Ending index of the slice. Default: 0.
        ellipsis_mask (int): An int mask. Default: 0.
        new_axis_mask (int): An int mask. Default: 0.
        shrink_axis_mask (int): An int mask. Default: 0.

    Inputs:
        - **input_x** (Tensor) - The input Tensor.
        - **begin** (tuple[int]) - A tuple which represents the location where to start. Only
          constant value is allowed.
        - **end** (tuple[int]) - A tuple or which represents the maximum location where to end.
          Only constant value is allowed.
        - **strides** (tuple[int]) - A tuple which represents the stride is continuously added
          before reaching the maximum location. Only constant value is allowed.

    Outputs:
        Tensor.
        The output is explained by following example.
            - In the 0th dimension, begin is 1, end is 2, and strides is 1,
              because :math:`1+1=2\geq2`, the interval is :math:`[1,2)`.
              Thus, return the element with :math:`index = 1` in 0th dimension, i.e., [[3, 3, 3], [4, 4, 4]].
            - In the 1st dimension, similarly, the interval is :math:`[0,1)`.
              Based on the return value of the 0th dimension, return the element with :math:`index = 0`,
              i.e., [3, 3, 3].
            - In the 2nd dimension, similarly, the interval is :math:`[0,3)`.
              Based on the return value of the 1st dimension, return the element with :math:`index = 0,1,2`,
              i.e., [3, 3, 3].
            - Finally, the output is [3, 3, 3].

    Examples
        >>> input_x = Tensor([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]],
        ...                   [[5, 5, 5], [6, 6, 6]]], mindspore.float32)
        >>> slice = P.StridedSlice()
        >>> output = slice(input_x, (1, 0, 0), (2, 1, 3), (1, 1, 1))
        >>> output.shape
        (1, 1, 3)
        >>> output
        [[[3, 3, 3]]]
    """

    @prim_attr_register
    def __init__(self,
                 begin_mask=0,
                 end_mask=0,
                 ellipsis_mask=0,
                 new_axis_mask=0,
                 shrink_axis_mask=0):
        """Initialize StrideSlice"""
        self.init_prim_io_names(inputs=['x', 'begin', 'end', 'strides'], outputs=['output'])
        validator.check_non_negative_int(begin_mask, 'begin_mask', self.name)
        validator.check_non_negative_int(end_mask, 'end_mask', self.name)
        validator.check_non_negative_int(ellipsis_mask, 'ellipsis_mask', self.name)
        if len(tuple(filter(lambda x: x == '1', bin(ellipsis_mask)[-1:1:-1]))) > 1:
            raise ValueError(f"For '{self.name}', only support one ellipsis in the index, but got {end_mask}.")
        validator.check_non_negative_int(new_axis_mask, 'new_axis_mask', self.name)
        validator.check_non_negative_int(shrink_axis_mask, 'shrink_axis_mask', self.name)

    def __infer__(self, x, begin, end, strides):
        begin_v, end_v, strides_v = begin['value'], end['value'], strides['value']
        validator.check_value_type("begin", begin_v, [tuple], self.name)
        validator.check_value_type("end", end_v, [tuple], self.name)
        validator.check_value_type("strides", strides_v, [tuple], self.name)

        if tuple(filter(lambda x: not isinstance(x, int), begin_v + end_v + strides_v)):
            raise TypeError(f"For {self.name}, both the begins, ends, and strides must be a tuple of int, "
                            f"but got begins: {begin_v}, ends: {end_v}, strides: {strides_v}.")

        if tuple(filter(lambda x: x == 0, strides_v)):
            raise ValueError(f"For '{self.name}', the strides cannot contain 0, but got strides: {strides_v}.")

        if len(end_v) != len(begin_v) or len(strides_v) != len(begin_v):
            raise ValueError(f"For '{self.name}' the length of begin index: {begin_v}, end index: {end_v} and "
                             f"strides: {strides_v} must be equal.")

        ret_shape = self._compute_slicing_shape(x['shape'], begin_v, end_v, strides_v)

        value = None if all(ret_shape) else Tensor(np.array([]).reshape(ret_shape), x['dtype'].element_type())
        return {'shape': ret_shape,
                'dtype': x['dtype'],
                'value': value}

    def _compute_slicing_shape(self, x_shape, begin_v, end_v, strides_v):
        """Computes the shape of the slicing."""
        x_rank = len(x_shape)
        slice_len = len(begin_v)

        # After the integer is converted to binary, it is a str and the first two chars are the flag char '0b'.
        begin_pos = bin(self.begin_mask)[-1:1:-1]
        end_pos = bin(self.end_mask)[-1:1:-1]
        ellipsis_pos = bin(self.ellipsis_mask)[-1:1:-1]
        new_axis_pos = bin(self.new_axis_mask)[-1:1:-1]
        shrink_axis_pos = bin(self.shrink_axis_mask)[-1:1:-1]

        ret_shape = []
        i, j = 0, 0
        has_ellipsis = False
        while i < x_rank or j < slice_len:
            if j < slice_len:
                begin, end, stride = begin_v[j], end_v[j], strides_v[j]

                if j < len(ellipsis_pos) and ellipsis_pos[j] == '1':
                    # When there is ellipsis, the latter part of the ellipsis will be processed separately.
                    has_ellipsis = True
                    break
                if j < len(begin_pos) and begin_pos[j] == '1':
                    begin = -1 if strides_v[j] < 0 else 0
                if j < len(end_pos) and end_pos[j] == '1':
                    end = -(x_shape[i] + 1) if strides_v[j] < 0 else x_shape[i]
                if j < len(new_axis_pos) and new_axis_pos[j] == '1':
                    ret_shape.append(1)
                    j += 1
                    continue
                if j < len(shrink_axis_pos) and shrink_axis_pos[j] == '1':
                    if (not -x_shape[i] <= begin < x_shape[i]) or stride < 0:
                        raise ValueError(f"For {self.name}, when shrink axis, the stride cannot be negative number, "
                                         f"and begin should be in [-{x_shape[i]}, {x_shape[i]}), "
                                         f"but got stride: {stride}, begin: {begin}.")
                    j += 1
                    i += 1
                    continue
            else:
                begin, end, stride = 0, x_shape[i], 1

            slicing_length = _compute_slicing_length(begin, end, stride, x_shape, i)
            ret_shape.append(slicing_length)
            i += 1
            j += 1
        if has_ellipsis:
            # When there is ellipsis, handle the second half of the ellipsis split.
            ellipsis_occupied_dims = x_rank - i - (slice_len - (j + 1)) + \
                                     len(tuple(filter(lambda x: x == '1', new_axis_pos[j + 1:slice_len])))
            ret_shape.extend(x_shape[i:i + ellipsis_occupied_dims])
            j += 1
            i += ellipsis_occupied_dims

            while i < x_rank or j < slice_len:
                begin, end, stride = begin_v[j], end_v[j], strides_v[j]

                if j < len(begin_pos) and begin_pos[j] == '1':
                    begin = -1 if strides_v[j] < 0 else 0
                if j < len(end_pos) and end_pos[j] == '1':
                    end = -(x_shape[i] + 1) if strides_v[j] < 0 else x_shape[i]
                if j < len(new_axis_pos) and new_axis_pos[j] == '1':
                    ret_shape.append(1)
                    j += 1
                    continue
                if j < len(shrink_axis_pos) and shrink_axis_pos[j] == '1':
                    if (not -x_shape[i] <= begin < x_shape[i]) or stride < 0:
                        raise ValueError(f"For {self.name}, when shrink axis, the stride cannot be negative number, "
                                         f"and begin should be in [-{x_shape[i]}, {x_shape[i]}), "
                                         f"but got stride: {stride}, begin: {begin}.")
                    j += 1
                    i += 1
                    continue

                slicing_length = _compute_slicing_length(begin, end, stride, x_shape, i)
                ret_shape.append(slicing_length)
                i += 1
                j += 1
        return ret_shape


class Diag(PrimitiveWithInfer):
    r"""

    Constructs a diagonal tensor with a given diagonal values.

    Assume `input_x` has dimensions :math:`[D_1,... D_k]`, the output is a tensor of
    rank 2k with dimensions :math:`[D_1,..., D_k, D_1,..., D_k]` where:
    :math:`output[i_1,..., i_k, i_1,..., i_k] = input_x[i_1,..., i_k]` and 0 everywhere else.

    Inputs:
        - **input_x** (Tensor) - The input tensor. The input shape must be less than 5d.

    Outputs:
        Tensor, has the same dtype as the `input_x`.

    Examples:
        >>> input_x = Tensor([1, 2, 3, 4])
        >>> diag = P.Diag()
        >>> diag(input_x)
        [[1, 0, 0, 0],
         [0, 2, 0, 0],
         [0, 0, 3, 0],
         [0, 0, 0, 4]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Diag"""

    def infer_dtype(self, x_type):
        validator.check_subclass('input_x', x_type, mstype.tensor, self.name)
        return x_type

    def infer_shape(self, x_shape):
        validator.check("x rank", len(x_shape), "", 1, Rel.GE)
        ret_shape = copy.deepcopy(x_shape)
        ret_shape = ret_shape + ret_shape
        return ret_shape

    def infer_value(self, x):
        if x is None:
            return None
        # do constant-folding only when x rank is 1
        if len(x.shape) != 1:
            return None
        ret = np.diag(x.asnumpy())
        return Tensor(ret)


class DiagPart(PrimitiveWithInfer):
    r"""

    Extracts the diagonal part from given tensor.

    Assume input has dimensions :math:`[D_1,..., D_k, D_1,..., D_k]`, the output is a tensor
    of rank k with dimensions :math:`[D_1,..., D_k]` where:
    :math:`output[i_1,..., i_k] = input[i_1,..., i_k, i_1,..., i_k]`.

    Inputs:
        - **input_x** (Tensor) - tensor of rank k where k is even and not zero.

    Outputs:
        Tensor, the extracted diagonal has the same dtype as the `input_x`.

    Examples
        >>> input_x = Tensor([[1, 0, 0, 0],
        >>>                   [0, 2, 0, 0],
        >>>                   [0, 0, 3, 0],
        >>>                   [0, 0, 0, 4]])
        >>> diag_part = P.DiagPart()
        >>> output = diag_part(input_x)
        >>> print(output)
        [1, 2, 3, 4]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize DiagPart"""

    def infer_dtype(self, x_type):
        validator.check_subclass('input_x', x_type, mstype.tensor, self.name)
        return x_type

    def infer_shape(self, x_shape):
        if len(x_shape) % 2 != 0 or \
                not x_shape:
            raise ValueError(f"For \'{self.name}\' input rank must be non-zero and even, but got rank {len(x_shape)}, "
                             f"with shapes {x_shape}")
        length = len(x_shape) // 2
        for i in range(length):
            validator.check('input_shape[i + len(input_shape)/2]', x_shape[i + length],
                            'input_shape[i]', x_shape[i], Rel.EQ, self.name)
        ret_shape = x_shape[0:length]
        return ret_shape

    def infer_value(self, x):
        if x is None:
            return None
        # do constant-folding only when x rank is 2
        if len(x.shape) != 2:
            return None
        ret = np.diag(x.asnumpy())
        return Tensor(ret)


class Eye(PrimitiveWithInfer):
    """

    Creates a tensor with ones on the diagonal and zeros the rest.

    Inputs:
        - **n** (int) - The number of rows of returned tensor
        - **m** (int) - The number of columns of returned tensor
        - **t** (mindspore.dtype) - MindSpore's dtype, The data type of the returned tensor.

    Outputs:
        Tensor, a tensor with ones on the diagonal and the rest of elements are zero.

    Examples:
        >>> eye = P.Eye()
        >>> out_tensor = eye(2, 2, mindspore.int32)
        >>> print(out_tensor)
        [[1, 0],
         [0, 1]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Eye"""

    def infer_value(self, n, m, t):
        validator.check_positive_int(n, "n", self.name)
        validator.check_positive_int(m, "m", self.name)
        args = {"dtype": t}
        validator.check_types_same_and_valid(args, mstype.number_type + (mstype.bool_,), self.name)
        np_type = mstype.dtype_to_nptype(t)
        ret = np.eye(n, m, dtype=np_type)
        return Tensor(ret)


class ScatterNd(PrimitiveWithInfer):
    """
    Scatters a tensor into a new tensor depending on the specified indices.

    Creates an empty tensor, and set values by scattering the update tensor depending on indices.

    Inputs:
        - **indices** (Tensor) - The index of scattering in the new tensor with int32 data type.
        - **update** (Tensor) - The source Tensor to be scattered.
        - **shape** (tuple[int]) - Define the shape of the output tensor, has the same type as indices.

    Outputs:
        Tensor, the new tensor, has the same type as `update` and the same shape as `shape`.

    Examples:
        >>> op = P.ScatterNd()
        >>> indices = Tensor(np.array([[0, 1], [1, 1]]), mindspore.int32)
        >>> update = Tensor(np.array([3.2, 1.1]), mindspore.float32)
        >>> shape = (3, 3)
        >>> output = op(indices, update, shape)
        >>> print(output)
        [[0. 3.2 0.]
         [0. 1.1 0.]
         [0. 0. 0. ]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize ScatterNd"""
        self.init_prim_io_names(inputs=['indices', 'update', 'shape'], outputs=['output'])

    def __infer__(self, indices, update, shape):
        shp = shape['value']
        validator.check_subclass("update_dtype", update['dtype'], mstype.tensor, self.name)
        validator.check_tensor_dtype_valid("indices", indices['dtype'], [mstype.int32], self.name)
        validator.check_value_type("shape", shp, [tuple], self.name)
        for i, x in enumerate(shp):
            validator.check_positive_int(x, f'shape[{i}]', self.name)

        indices_shape, update_shape = indices["shape"], update["shape"]
        if indices_shape[0] != update_shape[0]:
            raise ValueError(f'For \'{self.name}\' The indices_shape[0] and update_shape[0] must be equal.')

        return {'shape': shp,
                'dtype': update['dtype'],
                'value': None}


class ResizeNearestNeighbor(PrimitiveWithInfer):
    r"""
    Resizes the input tensor by using nearest neighbor algorithm.

    Resizes the input tensor to a given size by using the nearest neighbor algorithm. The nearest
    neighbor algorithm selects the value of the nearest point and does not consider the
    values of neighboring points at all, yielding a piecewise-constant interpolant.

    Args:
        size (Union[tuple, list]): The target size. The dimension of size must be 2.
        align_corners (bool): Whether the centers of the 4 corner pixels of the input
                              and output tensors are aligned. Default: False.

    Inputs:
        - **input_x** (Tensor) - The input tensor. The shape of the tensor is :math:`(N, C, H, W)`.

    Outputs:
        Tensor, the shape of the output tensor is :math:`(N, C, NEW\_H, NEW\_W)`.

    Examples:
        >>> input_tensor = Tensor(np.array([[[[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]]]), mindspore.float32)
        >>> resize = P.ResizeNearestNeighbor((2, 2))
        >>> output = resize(input_tensor)
        >>> print(output)
        [[[[-0.1 0.3]
           [0.4 0.5 ]]]]
    """

    @prim_attr_register
    def __init__(self, size, align_corners=False):
        """Initialize ResizeNearestNeighbor"""
        validator.check_value_type("size", size, [tuple, list], self.name)
        validator.check_value_type("align_corners", align_corners, [bool], self.name)
        validator.check_equal_int(len(size), 2, "length of size", self.name)
        for i, value in enumerate(size):
            validator.check_non_negative_int(value, f'{i}th value of size', self.name)
        self.init_prim_io_names(inputs=['image_in'], outputs=['image_out'])

    def infer_shape(self, x_shape):
        validator.check('the dimension of input_x', len(x_shape), '', 4, Rel.EQ, self.name)
        return tuple(x_shape)[:-2] + tuple(self.size)

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid("x", x_dtype, mstype.number_type, self.name)
        return x_dtype


class GatherNd(PrimitiveWithInfer):
    """
    Gathers slices from a tensor by indices.

    Using given indices to gather slices from a tensor with a specified shape.

    Inputs:
        - **input_x** (Tensor) - The target tensor to gather values.
        - **indices** (Tensor) - The index tensor, with int data type.

    Outputs:
        Tensor, has the same type as `input_x` and the shape is indices_shape[:-1] + x_shape[indices_shape[-1]:].

    Examples:
        >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
        >>> indices = Tensor(np.array([[0, 0], [1, 1]]), mindspore.int32)
        >>> op = P.GatherNd()
        >>> output = op(input_x, indices)
        >>> print(output)
        [-0.1, 0.5]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize GatherNd"""
        self.init_prim_io_names(inputs=['input_x', 'indices'], outputs=['y'])

    def infer_shape(self, x_shape, indices_shape):
        validator.check('the dimension of x', len(x_shape),
                        'the dimension of indices', indices_shape[-1], Rel.GE, self.name)
        return indices_shape[:-1] + x_shape[indices_shape[-1]:]

    def infer_dtype(self, x_dtype, indices_dtype):
        validator.check_tensor_dtype_valid("indices", indices_dtype, mstype.int_type, self.name)
        return x_dtype


class TensorScatterUpdate(PrimitiveWithInfer):
    """
    Updates tensor value using given values, along with the input indices.

    Inputs:
        - **input_x** (Tensor) - The target tensor. The dimension of input_x must be equal to indices.shape[-1].
        - **indices** (Tensor) - The index of input tensor whose data type is int32.
        - **update** (Tensor) - The tensor to update the input tensor, has the same type as input,
          and update.shape = indices.shape[:-1] + input_x.shape[indices.shape[-1]:].

    Outputs:
        Tensor, has the same shape and type as `input_x`.

    Examples:
        >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
        >>> indices = Tensor(np.array([[0, 0], [1, 1]]), mindspore.int32)
        >>> update = Tensor(np.array([1.0, 2.2]), mindspore.float32)
        >>> op = P.TensorScatterUpdate()
        >>> output = op(input_x, indices, update)
        [[1.0, 0.3, 3.6],
         [0.4, 2.2, -3.2]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize TensorScatterUpdate"""
        self.init_prim_io_names(inputs=['x', 'indices', 'value'], outputs=['y'])

    def infer_shape(self, x_shape, indices_shape, value_shape):
        validator.check('the dimension of x', len(x_shape),
                        'the dimension of indices', indices_shape[-1], Rel.GE)
        if indices_shape[:-1] + x_shape[indices_shape[-1]:] != value_shape:
            raise ValueError("For 'TensorScatterUpdate', input value are not match with input indices.")
        return x_shape

    def infer_dtype(self, x_dtype, indices_dtype, value_dtype):
        validator.check_tensor_dtype_valid('indices', indices_dtype, [mstype.int32], self.name)
        args = {"x": x_dtype, "value": value_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, (mstype.bool_,) + mstype.number_type, self.name)
        return x_dtype


class ScatterUpdate(_ScatterOp_Dynamic):
    """
    Updates tensor value by using input indices and value.

    Using given values to update tensor value, along with the input indices.

    Inputs of `input_x` and `updates` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Args:
        use_locking (bool): Whether protect the assignment by a lock. Default: True.

    Inputs:
        - **input_x** (Parameter) - The target tensor, with data type of Parameter.
        - **indices** (Tensor) - The index of input tensor. With int32 data type.
        - **updates** (Tensor) - The tensor to update the input tensor, has the same type as input,
          and updates.shape = indices.shape + input_x.shape[1:].

    Outputs:
        Tensor, has the same shape and type as `input_x`.

    Examples:
        >>> np_x = np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]])
        >>> input_x = mindspore.Parameter(Tensor(np_x, mindspore.float32), name="x")
        >>> indices = Tensor(np.array([[0, 0], [1, 1]]), mindspore.int32)
        >>> np_updates = np.array([[[1.0, 2.2, 1.0], [2.0, 1.2, 1.0]], [[2.0, 2.2, 1.0], [3.0, 1.2, 1.0]]])
        >>> updates = Tensor(np_updates, mindspore.float32)
        >>> op = P.ScatterUpdate()
        >>> output = op(input_x, indices, updates)
        >>> print(output)
        [[2.0, 1.2, 1.0],
         [3.0, 1.2, 1.0]]
    """
    @prim_attr_register
    def __init__(self, use_locking=True):
        """Initialize ScatterUpdate"""
        validator.check_value_type('use_locking', use_locking, [bool], self.name)
        self.init_prim_io_names(inputs=['x', 'indices', 'updates'], outputs=['y'])

class ScatterNdUpdate(_ScatterNdOp):
    """
    Updates tensor value by using input indices and value.

    Using given values to update tensor value, along with the input indices.

    Inputs of `input_x` and `updates` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Args:
        use_locking (bool): Whether protect the assignment by a lock. Default: True.

    Inputs:
        - **input_x** (Parameter) - The target tensor, with data type of Parameter.
        - **indices** (Tensor) - The index of input tensor, with int32 data type.
        - **update** (Tensor) - The tensor to be updated to the input tensor, has the same type as input.

    Outputs:
        Tensor, has the same shape and type as `input_x`.

    Examples:
        >>> np_x = np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]])
        >>> input_x = mindspore.Parameter(Tensor(np_x, mindspore.float32), name="x")
        >>> indices = Tensor(np.array([[0, 0], [1, 1]]), mindspore.int32)
        >>> update = Tensor(np.array([1.0, 2.2]), mindspore.float32)
        >>> op = P.ScatterNdUpdate()
        >>> output = op(input_x, indices, update)
        >>> print(output)
        [[1. 0.3 3.6]
         [0.4 2.2 -3.2]]
    """

    @prim_attr_register
    def __init__(self, use_locking=True):
        """Initialize ScatterNdUpdate"""
        validator.check_value_type('use_locking', use_locking, [bool], self.name)
        self.init_prim_io_names(inputs=['x', 'indices', 'value'], outputs=['y'])

    def infer_dtype(self, x_dtype, indices_dtype, value_dtype):
        validator.check_tensor_dtype_valid('indices', indices_dtype, [mstype.int32], self.name)
        args = {"x": x_dtype, "value": value_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, (mstype.bool_,) + mstype.number_type, self.name)
        return x_dtype


class ScatterMax(_ScatterOp):
    """
    Updates the value of the input tensor through the max operation.

    Using given values to update tensor value through the max operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    Inputs of `input_x` and `updates` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Args:
        use_locking (bool): Whether protect the assignment by a lock. Default: True.

    Inputs:
        - **input_x** (Parameter) - The target parameter.
        - **indices** (Tensor) - The index to do max operation whose data type must be mindspore.int32.
        - **updates** (Tensor) - The tensor that performs the maximum operation with `input_x`,
          the data type is the same as `input_x`, the shape is `indices_shape + x_shape[1:]`.

    Outputs:
        Parameter, the updated `input_x`.

    Examples:
        >>> input_x = Parameter(Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mindspore.float32), name="input_x")
        >>> indices = Tensor(np.array([[0, 0], [1, 1]]), mindspore.int32)
        >>> update = Tensor(np.ones([2, 2, 3]) * 88, mindspore.float32)
        >>> scatter_max = P.ScatterMax()
        >>> output = scatter_max(input_x, indices, update)
        >>> print(output)
        [[88.0, 88.0, 88.0], [88.0, 88.0, 88.0]]
    """

    @prim_attr_register
    def __init__(self, use_locking=True):
        """Initialize ScatterMax"""
        self.init_prim_io_names(inputs=['x', 'indices', 'updates'], outputs=['y'])
        validator.check_value_type('use_locking', use_locking, (bool,), self.name)


class ScatterMin(_ScatterOp):
    """
    Updates the value of the input tensor through the min operation.

    Using given values to update tensor value through the min operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    Inputs of `input_x` and `updates` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Args:
        use_locking (bool): Whether protect the assignment by a lock. Default: False.

    Inputs:
        - **input_x** (Parameter) - The target parameter.
        - **indices** (Tensor) - The index to do min operation whose data type must be mindspore.int32.
        - **updates** (Tensor) - The tensor doing the min operation with `input_x`,
          the data type is same as `input_x`, the shape is `indices_shape + x_shape[1:]`.

    Outputs:
        Parameter, the updated `input_x`.

    Examples:
        >>> input_x = Parameter(Tensor(np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]]), mindspore.float32), name="input_x")
        >>> indices = Tensor(np.array([[0, 0], [1, 1]]), mindspore.int32)
        >>> update = Tensor(np.ones([2, 2, 3]), mindspore.float32)
        >>> scatter_min = P.ScatterMin()
        >>> output = scatter_min(input_x, indices, update)
        >>> print(output)
        [[0.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
    """


class ScatterAdd(_ScatterOp_Dynamic):
    """
    Updates the value of the input tensor through the add operation.

    Using given values to update tensor value through the add operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    Inputs of `input_x` and `updates` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Args:
        use_locking (bool): Whether protect the assignment by a lock. Default: False.

    Inputs:
        - **input_x** (Parameter) - The target parameter.
        - **indices** (Tensor) - The index to do add operation whose data type must be mindspore.int32.
        - **updates** (Tensor) - The tensor that performs the add operation with `input_x`,
          the data type is the same as `input_x`, the shape is `indices_shape + x_shape[1:]`.

    Outputs:
        Parameter, the updated `input_x`.

    Examples:
        >>> input_x = Parameter(Tensor(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), mindspore.float32), name="x")
        >>> indices = Tensor(np.array([[0, 1], [1, 1]]), mindspore.int32)
        >>> updates = Tensor(np.ones([2, 2, 3]), mindspore.float32)
        >>> scatter_add = P.ScatterAdd()
        >>> output = scatter_add(input_x, indices, updates)
        >>> print(output)
        [[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]]
    """
    @prim_attr_register
    def __init__(self, use_locking=False):
        """Initialize ScatterAdd"""
        validator.check_value_type('use_locking', use_locking, [bool], self.name)
        self.init_prim_io_names(inputs=['x', 'indices', 'updates'], outputs=['y'])


class ScatterSub(_ScatterOp):
    """
    Updates the value of the input tensor through the subtraction operation.

    Using given values to update tensor value through the subtraction operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    Inputs of `input_x` and `updates` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Args:
        use_locking (bool): Whether protect the assignment by a lock. Default: False.

    Inputs:
        - **input_x** (Parameter) - The target parameter.
        - **indices** (Tensor) - The index to perform the subtraction operation
          whose data type must be mindspore.int32.
        - **updates** (Tensor) - The tensor that performs the subtraction operation with `input_x`,
          the data type is the same as `input_x`, the shape is `indices_shape + x_shape[1:]`.

    Outputs:
        Parameter, the updated `input_x`.

    Examples:
        >>> input_x = Parameter(Tensor(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]), mindspore.float32), name="x")
        >>> indices = Tensor(np.array([[0, 1]]), mindspore.int32)
        >>> updates = Tensor(np.array([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]]), mindspore.float32)
        >>> scatter_sub = P.ScatterSub()
        >>> output = scatter_sub(input_x, indices, updates)
        >>> print(output)
        [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]
    """


class ScatterMul(_ScatterOp):
    """
    Updates the value of the input tensor through the mul operation.

    Using given values to update tensor value through the mul operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    Inputs of `input_x` and `updates` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Args:
        use_locking (bool): Whether protect the assignment by a lock. Default: False.

    Inputs:
        - **input_x** (Parameter) - The target parameter.
        - **indices** (Tensor) - The index to do mul operation whose data type must be mindspore.int32.
        - **updates** (Tensor) - The tensor doing the mul operation with `input_x`,
          the data type is same as `input_x`, the shape is `indices_shape + x_shape[1:]`.

    Outputs:
        Parameter, the updated `input_x`.

    Examples:
        >>> input_x = Parameter(Tensor(np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]), mindspore.float32), name="x")
        >>> indices = Tensor(np.array([0, 1]), mindspore.int32)
        >>> updates = Tensor(np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]), mindspore.float32)
        >>> scatter_mul = P.ScatterMul()
        >>> output = scatter_mul(input_x, indices, updates)
        >>> print(output)
        [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]
    """


class ScatterDiv(_ScatterOp):
    """
    Updates the value of the input tensor through the div operation.

    Using given values to update tensor value through the div operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    Inputs of `input_x` and `updates` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Args:
        use_locking (bool): Whether protect the assignment by a lock. Default: False.

    Inputs:
        - **input_x** (Parameter) - The target parameter.
        - **indices** (Tensor) - The index to do div operation whose data type must be mindspore.int32.
        - **updates** (Tensor) - The tensor that performs the div operation with `input_x`,
          the data type is the same as `input_x`, the shape is `indices_shape + x_shape[1:]`.

    Outputs:
        Parameter, the updated `input_x`.

    Examples:
        >>> input_x = Parameter(Tensor(np.array([[6.0, 6.0, 6.0], [2.0, 2.0, 2.0]]), mindspore.float32), name="x")
        >>> indices = Tensor(np.array([0, 1]), mindspore.int32)
        >>> updates = Tensor(np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]), mindspore.float32)
        >>> scatter_div = P.ScatterDiv()
        >>> output = scatter_div(input_x, indices, updates)
        >>> print(output)
        [[3.0, 3.0, 3.0], [1.0, 1.0, 1.0]]
    """


class ScatterNdAdd(_ScatterNdOp):
    """
    Applies sparse addition to individual values or slices in a Tensor.

    Using given values to update tensor value through the add operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    Inputs of `input_x` and `updates` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Args:
        use_locking (bool): Whether protect the assignment by a lock. Default: False.

    Inputs:
        - **input_x** (Parameter) - The target parameter.
        - **indices** (Tensor) - The index to do add operation whose data type must be mindspore.int32.
        - **updates** (Tensor) - The tensor doing the add operation with `input_x`,
          the data type is same as `input_x`, the shape is `indices_shape[:-1] + x_shape[indices_shape[-1]:]`.

    Outputs:
        Parameter, the updated `input_x`.

    Examples:
        >>> input_x = Parameter(Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8]), mindspore.float32), name="x")
        >>> indices = Tensor(np.array([[2], [4], [1], [7]]), mindspore.int32)
        >>> updates = Tensor(np.array([6, 7, 8, 9]), mindspore.float32)
        >>> scatter_nd_add = P.ScatterNdAdd()
        >>> output = scatter_nd_add(input_x, indices, updates)
        >>> print(output)
        [1, 10, 9, 4, 12, 6, 7, 17]
    """


class ScatterNdSub(_ScatterNdOp):
    """
    Applies sparse subtraction to individual values or slices in a Tensor.

    Using given values to update tensor value through the subtraction operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    Inputs of `input_x` and `updates` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Args:
        use_locking (bool): Whether protect the assignment by a lock. Default: False.

    Inputs:
        - **input_x** (Parameter) - The target parameter.
        - **indices** (Tensor) - The index to do add operation whose data type must be mindspore.int32.
        - **updates** (Tensor) - The tensor that performs the subtraction operation with `input_x`,
          the data type is the same as `input_x`, the shape is `indices_shape[:-1] + x_shape[indices_shape[-1]:]`.

    Outputs:
        Parameter, the updated `input_x`.

    Examples:
        >>> input_x = Parameter(Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8]), mindspore.float32), name="x")
        >>> indices = Tensor(np.array([[2], [4], [1], [7]]), mindspore.int32)
        >>> updates = Tensor(np.array([6, 7, 8, 9]), mindspore.float32)
        >>> scatter_nd_sub = P.ScatterNdSub()
        >>> output = scatter_nd_sub(input_x, indices, updates)
        >>> print(output)
        [1, -6, -3, 4, -2, 6, 7, -1]
    """


class ScatterNonAliasingAdd(_ScatterNdOp):
    """
    Applies sparse addition to input using individual values or slices.

    Using given values to update tensor value through the add operation, along with the input indices.
    This operation outputs the `input_x` after the update is done, which makes it convenient to use the updated value.

    Inputs of `input_x` and `updates` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Inputs:
        - **input_x** (Parameter) - The target parameter. The data type must be float16, float32 or int32.
        - **indices** (Tensor) - The index to perform the addition operation whose data type must be mindspore.int32.
        - **updates** (Tensor) - The tensor that performs the addition operation with `input_x`,
          the data type is the same as `input_x`, the shape is `indices_shape[:-1] + x_shape[indices_shape[-1]:]`.

    Outputs:
        Parameter, the updated `input_x`.

    Examples:
        >>> input_x = Parameter(Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8]), mindspore.float32), name="x")
        >>> indices = Tensor(np.array([[2], [4], [1], [7]]), mindspore.int32)
        >>> updates = Tensor(np.array([6, 7, 8, 9]), mindspore.float32)
        >>> scatter_non_aliasing_add = P.ScatterNonAliasingAdd()
        >>> output = scatter_non_aliasing_add(input_x, indices, updates)
        >>> print(output)
        [1, 10, 9, 4, 12, 6, 7, 17]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize ScatterNonAliasingAdd"""
        self.init_prim_io_names(inputs=['x', 'indices', 'updates'], outputs=['y'])

    def infer_dtype(self, x_dtype, indices_dtype, updates_dtype):
        validator.check_tensor_dtype_valid('indices', indices_dtype, [mstype.int32], self.name)
        args = {"x": x_dtype, "updates": updates_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, [mstype.float16, mstype.float32, mstype.int32], self.name)
        return x_dtype


class SpaceToDepth(PrimitiveWithInfer):
    r"""
    Rearranges blocks of spatial data into depth.

    The output tensor's `height` dimension is :math:`height / block\_size`.

    The output tensor's `weight` dimension is :math:`weight / block\_size`.

    The depth of output tensor is :math:`block\_size * block\_size * input\_depth`.

    The input tensor's height and width must be divisible by `block_size`.
    The data format is "NCHW".

    Args:
        block_size (int): The block size used to divide spatial data. It must be >= 2.

    Inputs:
        - **x** (Tensor) - The target tensor.

    Outputs:
        Tensor, the same data type as `x`. It must be a 4-D tensor.

    Examples:
        >>> x = Tensor(np.random.rand(1,3,2,2), mindspore.float32)
        >>> block_size = 2
        >>> op = P.SpaceToDepth(block_size)
        >>> output = op(x)
        >>> output.asnumpy().shape == (1,12,1,1)
    """

    @prim_attr_register
    def __init__(self, block_size):
        """Initialize SpaceToDepth"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])
        validator.check_value_type('block_size', block_size, [int], self.name)
        validator.check('block_size', block_size, '', 2, Rel.GE)
        self.block_size = block_size

    def infer_shape(self, x_shape):
        validator.check('x dimension', len(x_shape), '', 4, Rel.EQ)
        out_shape = copy.deepcopy(x_shape)
        for i in range(2):
            if out_shape[i + 2] % self.block_size != 0:
                raise ValueError(f'For \'{self.name}\' input shape[{i + 2}] {out_shape[i + 2]} should be '
                                 f'fully divided by block_size {self.block_size}')
            out_shape[i + 2] //= self.block_size

        out_shape[1] *= self.block_size * self.block_size
        return out_shape

    def infer_dtype(self, x_dtype):
        validator.check_subclass("x_dtype", x_dtype, mstype.tensor, self.name)
        return x_dtype


class DepthToSpace(PrimitiveWithInfer):
    r"""
    Rearranges blocks of depth data into spatial dimensions.

    This is the reverse operation of SpaceToDepth.

    The depth of output tensor is :math:`input\_depth / (block\_size * block\_size)`.

    The output tensor's `height` dimension is :math:`height * block\_size`.

    The output tensor's `weight` dimension is :math:`weight * block\_size`.

    The input tensor's depth must be divisible by `block_size * block_size`.
    The data format is "NCHW".

    Args:
        block_size (int): The block size used to divide depth data. It must be >= 2.

    Inputs:
        - **x** (Tensor) - The target tensor. It must be a 4-D tensor with shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{in} / \text{block_size}, H_{in} * \text{block_size}, W_{in} * \text{block_size})`.

    Examples:
        >>> x = Tensor(np.random.rand(1,12,1,1), mindspore.float32)
        >>> block_size = 2
        >>> op = P.DepthToSpace(block_size)
        >>> output = op(x)
        >>> print(output.shape)
        (1, 3, 2, 2)
    """

    @prim_attr_register
    def __init__(self, block_size):
        """Initialize DepthToSpace"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])
        validator.check_value_type('block_size', block_size, [int], self.name)
        validator.check('block_size', block_size, '', 2, Rel.GE, self.name)
        self.block_size = block_size

    def infer_shape(self, x_shape):
        validator.check('x dimension', len(x_shape), '', 4, Rel.EQ)
        out_shape = copy.deepcopy(x_shape)
        for i in range(2):
            out_shape[i + 2] *= self.block_size

        validator.check_int(x_shape[1] % (self.block_size * self.block_size),
                            0, Rel.EQ, 'x_shape[1] % (block_size*block_size)', self.name)
        out_shape[1] //= self.block_size * self.block_size
        return out_shape

    def infer_dtype(self, x_dtype):
        validator.check_subclass("x_dtype", x_dtype, mstype.tensor, self.name)
        return x_dtype


class SpaceToBatch(PrimitiveWithInfer):
    r"""
    Divides spatial dimensions into blocks and combine the block size with the original batch.

    This operation will divide spatial dimensions (H, W) into blocks with `block_size`, the output tensor's H and W
    dimension is the corresponding number of blocks after division. The output tensor's batch dimension is the
    product of the original batch and the square of block_size. Before division, the spatial dimensions
    of the input are zero padded according to paddings if necessary.

    Args:
        block_size (int): The block size of dividing blocks with value greater than 2.
        paddings (Union[tuple, list]): The padding values for H and W dimension, containing 2 subtraction lists.
            Each subtraction list contains 2 integer value. All values must be greater than 0.
            paddings[i] specifies the paddings for the spatial dimension i, which corresponds to the
            input dimension i+2. It is required that input_shape[i+2]+paddings[i][0]+paddings[i][1]
            is divisible by block_size.

    Inputs:
        - **input_x** (Tensor) - The input tensor. It must be a 4-D tensor.

    Outputs:
        Tensor, the output tensor with the same data type as input. Assume input shape is :math:`(n, c, h, w)` with
        :math:`block\_size` and :math:`paddings`. The shape of the output tensor will be :math:`(n', c', h', w')`,
        where

            :math:`n' = n*(block\_size*block\_size)`

            :math:`c' = c`

            :math:`h' = (h+paddings[0][0]+paddings[0][1])//block\_size`

            :math:`w' = (w+paddings[1][0]+paddings[1][1])//block\_size`

    Examples:
        >>> block_size = 2
        >>> paddings = [[0, 0], [0, 0]]
        >>> space_to_batch = P.SpaceToBatch(block_size, paddings)
        >>> input_x = Tensor(np.array([[[[1, 2], [3, 4]]]]), mindspore.float32)
        >>> space_to_batch(input_x)
        [[[[1.]]], [[[2.]]], [[[3.]]], [[[4.]]]]

    """

    @prim_attr_register
    def __init__(self, block_size, paddings):
        """Initialize SpaceToBatch"""
        validator.check_value_type('block_size', block_size, [int], self.name)
        validator.check('block_size', block_size, '', 2, Rel.GE, self.name)
        self.block_size = block_size
        validator.check('paddings shape', np.array(paddings).shape, '', (2, 2), Rel.EQ, self.name)
        for elem in itertools.chain(*paddings):
            validator.check_non_negative_int(elem, 'paddings element', self.name)
            validator.check_value_type('paddings element', elem, [int], self.name)
        self.paddings = paddings

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('input_x', x_dtype, mstype.number_type, self.name)
        return x_dtype

    def infer_shape(self, x_shape):
        validator.check_equal_int(len(x_shape), 4, 'rank of input_x', self.name)
        out_shape = copy.deepcopy(x_shape)
        for i in range(2):
            padded = out_shape[i + 2] + self.paddings[i][0] + self.paddings[i][1]
            if padded % self.block_size != 0:
                raise ValueError(f'For \'{self.name}\' padded[{i}] {padded} should be divisible by '
                                 f'block_size {self.block_size}')
            out_shape[i + 2] = padded // self.block_size
        out_shape[0] *= self.block_size * self.block_size
        return out_shape


class BatchToSpace(PrimitiveWithInfer):
    r"""
    Divides batch dimension with blocks and interleaves these blocks back into spatial dimensions.

    This operation will divide batch dimension N into blocks with block_size, the output tensor's N dimension
    is the corresponding number of blocks after division. The output tensor's H, W dimension is product of original H, W
    dimension and block_size with given amount to crop from dimension, respectively.

    Args:
        block_size (int): The block size of division, has the value not less than 2.
        crops (Union[list(int), tuple(int)]): The crop value for H and W dimension, containing 2 subtraction lists.
            Each list contains 2 integers.
            All values must be not less than 0. crops[i] specifies the crop values for the spatial dimension i, which
            corresponds to the input dimension i+2. It is required that
            input_shape[i+2]*block_size >= crops[i][0]+crops[i][1].

    Inputs:
        - **input_x** (Tensor) - The input tensor. It must be a 4-D tensor, dimension 0 must be divisible by
          product of `block_shape`.

    Outputs:
        Tensor, the output tensor with the same type as input. Assume input shape is (n, c, h, w) with block_size
        and crops. The output shape will be (n', c', h', w'), where

                :math:`n' = n//(block\_size*block\_size)`

                :math:`c' = c`

                :math:`h' = h*block\_size-crops[0][0]-crops[0][1]`

                :math:`w' = w*block\_size-crops[1][0]-crops[1][1]`

    Examples:
        >>> block_size = 2
        >>> crops = [[0, 0], [0, 0]]
        >>> op = P.BatchToSpace(block_size, crops)
        >>> input_x = Tensor(np.array([[[[1]]], [[[2]]], [[[3]]], [[[4]]]]), mindspore.float32)
        >>> output = op(input_x)
        >>> print(output)
        [[[[1., 2.], [3., 4.]]]]

    """

    @prim_attr_register
    def __init__(self, block_size, crops):
        """Initialize BatchToSpace"""
        validator.check_value_type('block_size', block_size, [int], self.name)
        validator.check('block_size', block_size, '', 2, Rel.GE, self.name)
        self.block_size = block_size
        validator.check_value_type('crops type', crops, [list, tuple], self.name)
        validator.check('crops shape', np.array(crops).shape, '', (2, 2))
        for elem in itertools.chain(*crops):
            validator.check_non_negative_int(elem, 'crops element', self.name)
            validator.check_value_type('crops element', elem, [int], self.name)
        self.crops = crops

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('input_x', x_dtype, mstype.number_type, self.name)
        return x_dtype

    def infer_shape(self, x_shape):
        validator.check('rank of input_x', len(x_shape), '', 4)
        out_shape = copy.deepcopy(x_shape)
        for i in range(2):
            x_block_prod = out_shape[i + 2] * self.block_size
            crops_sum = self.crops[i][0] + self.crops[i][1]
            validator.check("x block shape prod", x_block_prod, 'crops sum', crops_sum, Rel.GT, self.name)
            out_shape[i + 2] = x_block_prod - crops_sum
        block_size_prod = self.block_size * self.block_size
        if out_shape[0] % block_size_prod != 0:
            raise ValueError(f'For \'{self.name}\' input_x dimension 0 {out_shape[0]}  should be divisible by '
                             f'block_size_prod {block_size_prod}')
        out_shape[0] = out_shape[0] // block_size_prod
        return out_shape


class SpaceToBatchND(PrimitiveWithInfer):
    r"""
    Divides spatial dimensions into blocks and combine the block size with the original batch.

    This operation will divide spatial dimensions (H, W) into blocks with block_shape, the output tensor's H and W
    dimension is the corresponding number of blocks after division. The output tensor's batch dimension is the
    product of the original batch and the product of `block_shape`. Before division,
    the spatial dimensions of the input are zero padded according to paddings if necessary.

    Args:
        block_shape (Union[list(int), tuple(int)]): The block shape of dividing block with all value greater than 1.
            The length of `block_shape` is M correspoding to the number of spatial dimensions. M must be 2.
        paddings (Union[tuple, list]): The padding values for H and W dimension, containing 2 subtraction list.
            Each contains 2 integer value. All values must be greater than 0.
            `paddings[i]` specifies the paddings for the spatial dimension i,
            which corresponds to the input dimension i+2.
            It is required that input_shape[i+2]+paddings[i][0]+paddings[i][1] is divisible by block_shape[i].

    Inputs:
        - **input_x** (Tensor) - The input tensor. It must be a 4-D tensor.
    Outputs:
        Tensor, the output tensor with the same data type as input. Assume input shape is :math:`(n, c, h, w)` with
        :math:`block\_shape` and :math:`padddings`. The shape of the output tensor will be :math:`(n', c', h', w')`,
        where

            :math:`n' = n*(block\_shape[0]*block\_shape[1])`

            :math:`c' = c`

            :math:`h' = (h+paddings[0][0]+paddings[0][1])//block\_shape[0]`

            :math:`w' = (w+paddings[1][0]+paddings[1][1])//block\_shape[1]`

    Examples:
        >>> block_shape = [2, 2]
        >>> paddings = [[0, 0], [0, 0]]
        >>> space_to_batch_nd = P.SpaceToBatchND(block_shape, paddings)
        >>> input_x = Tensor(np.array([[[[1, 2], [3, 4]]]]), mindspore.float32)
        >>> space_to_batch_nd(input_x)
        [[[[1.]]], [[[2.]]], [[[3.]]], [[[4.]]]]

    """

    @prim_attr_register
    def __init__(self, block_shape, paddings):
        """Initialize SpaceToBatchND"""
        self.ori_block_shape = block_shape
        self.ori_paddings = paddings
        validator.check_value_type('block_shape type', block_shape, [list, tuple], self.name)
        validator.check('block_shape shape', len(np.array(block_shape).shape), '', 1, Rel.EQ, self.name)
        block_rank = len(block_shape)
        validator.check('block_shape length', block_rank, '', 2, Rel.EQ, self.name)
        for elem in block_shape:
            validator.check('block_shape element', elem, '', 1, Rel.GE, self.name)
            validator.check_value_type('block_shape element', elem, [int], self.name)
        self.block_shape = block_shape

        validator.check_value_type('paddings type', paddings, [list, tuple], self.name)
        validator.check('paddings length', len(paddings), '', 2, Rel.EQ, self.name)
        validator.check('paddings shape', np.array(paddings).shape, '', (block_rank, 2), Rel.EQ, self.name)
        for elem in itertools.chain(*paddings):
            validator.check_non_negative_int(elem, 'paddings element', self.name)
            validator.check_value_type('paddings element', elem, [int], self.name)
        self.paddings = paddings
        block_shape_append = [1] + list(self.block_shape)
        self.add_prim_attr("block_shape", block_shape_append)
        paddings_append = [[0, 0]] + list(self.paddings)
        self.add_prim_attr("paddings", paddings_append)

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('input_x', x_dtype, mstype.number_type, self.name)
        return x_dtype

    def infer_shape(self, x_shape):
        x_rank = len(x_shape)
        validator.check_equal_int(x_rank, 4, 'x_shape rank', self.name)
        out_shape = copy.deepcopy(x_shape)

        block_shape_prod = 1
        offset = 2
        if x_rank <= 4:
            offset = 1
        for i in range(len(self.block_shape)):
            padded = out_shape[i + offset] + self.paddings[i][0] + \
                     self.paddings[i][1]
            if padded % self.block_shape[i] != 0:
                raise ValueError(f'For \'{self.name}\' padded[{i}] {padded} should be divisible by '
                                 f'block_shape[{i}] {self.block_shape[i]}')
            out_shape[i + offset] = padded // self.block_shape[i]
            block_shape_prod = block_shape_prod * self.block_shape[i]
        out_shape[0] *= block_shape_prod
        return out_shape


class BatchToSpaceND(PrimitiveWithInfer):
    r"""
    Divides batch dimension with blocks and interleave these blocks back into spatial dimensions.

    This operation will divide batch dimension N into blocks with block_shape, the output tensor's N dimension
    is the corresponding number of blocks after division. The output tensor's H, W dimension is product of original H, W
    dimension and block_shape with given amount to crop from dimension, respectively.

    Args:
        block_shape (Union[list(int), tuple(int)]): The block shape of dividing block with all value >= 1.
            The length of block_shape is M correspoding to the number of spatial dimensions. M must be 2.
        crops (Union[list(int), tuple(int)]): The crop value for H and W dimension, containing 2 subtraction list,
            each containing 2 int value.
            All values must be >= 0. crops[i] specifies the crop values for spatial dimension i, which corresponds to
            input dimension i+2. It is required that input_shape[i+2]*block_shape[i] > crops[i][0]+crops[i][1].

    Inputs:
        - **input_x** (Tensor) - The input tensor. It must be a 4-D tensor, dimension 0 must be divisible by
          product of `block_shape`.

    Outputs:
        Tensor, the output tensor with the same type as input. Assume input shape is (n, c, h, w) with block_shape
        and crops. The output shape will be (n', c', h', w'), where

                :math:`n' = n//(block\_shape[0]*block\_shape[1])`

                :math:`c' = c`

                :math:`h' = h*block\_shape[0]-crops[0][0]-crops[0][1]`

                :math:`w' = w*block\_shape[1]-crops[1][0]-crops[1][1]`

    Examples:
        >>> block_shape = [2, 2]
        >>> crops = [[0, 0], [0, 0]]
        >>> batch_to_space_nd = P.BatchToSpaceND(block_shape, crops)
        >>> input_x = Tensor(np.array([[[[1]]], [[[2]]], [[[3]]], [[[4]]]]), mindspore.float32)
        >>> output = batch_to_space_nd(input_x)
        >>> print(output)
        [[[[1., 2.], [3., 4.]]]]

    """

    @prim_attr_register
    def __init__(self, block_shape, crops):
        """Initialize BatchToSpaceND"""
        self.ori_block_shape = block_shape
        self.ori_crops = crops
        validator.check_value_type('block_shape type', block_shape, [list, tuple], self.name)
        validator.check('block_shape shape', len(np.array(block_shape).shape), '', 1, Rel.EQ, self.name)
        block_rank = len(block_shape)
        validator.check('block_shape length', block_rank, '', 2, Rel.EQ, self.name)
        for elem in block_shape:
            validator.check('block_shape element', elem, '', 1, Rel.GE, self.name)
            validator.check_value_type('block_shape element', elem, [int], self.name)
        self.block_shape = block_shape

        validator.check_value_type('crops type', crops, [list, tuple], self.name)
        validator.check('crops length', len(crops), '', 2, Rel.EQ, self.name)
        validator.check('crops shape', np.array(crops).shape, '', (block_rank, 2), Rel.EQ, self.name)
        for elem in itertools.chain(*crops):
            validator.check_non_negative_int(elem, 'crops element', self.name)
            validator.check_value_type('crops element', elem, [int], self.name)
        self.crops = crops
        block_shape_append = [1] + list(self.block_shape)
        self.add_prim_attr("block_shape", block_shape_append)
        crops_append = [[0, 0]] + list(self.crops)
        self.add_prim_attr("crops", crops_append)

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('input_x', x_dtype, mstype.number_type, self.name)
        return x_dtype

    def infer_shape(self, x_shape):
        x_rank = len(x_shape)
        validator.check_int(x_rank, 4, Rel.EQ, 'x_shape rank', self.name)
        out_shape = copy.deepcopy(x_shape)

        block_shape_prod = 1
        offset = 2
        if x_rank <= 4:
            offset = 1
        for i in range(len(self.block_shape)):
            block_shape_prod = block_shape_prod * self.block_shape[i]
            x_block_prod = out_shape[i + offset] * self.block_shape[i]
            crops_sum = self.crops[i][0] + self.crops[i][1]
            validator.check("x block shape prod", x_block_prod, 'crops sum', crops_sum, Rel.GT, self.name)
            out_shape[i + offset] = x_block_prod - crops_sum

        if out_shape[0] % block_shape_prod != 0:
            raise ValueError(f'For \'{self.name}\' input_x dimension 0 {out_shape[0]} should be divisible by '
                             f'block_shape_prod {block_shape_prod}')
        out_shape[0] = out_shape[0] // block_shape_prod
        return out_shape


class BroadcastTo(PrimitiveWithInfer):
    """
    Broadcasts input tensor to a given shape.
    Input shape can be broadcast to target shape if for each dimension pair they are either equal or input is one.
    When input shape is broadcast to target shape, it starts with the trailing dimensions.

    Args:
        shape (tuple): The target shape to broadcast.

    Inputs:
        - **input_x** (Tensor) - The input tensor.

    Outputs:
        Tensor, with the given `shape` and the same data type as `input_x`.

    Examples:
        >>> shape = (2, 3)
        >>> input_x = Tensor(np.array([1, 2, 3]).astype(np.float32))
        >>> broadcast_to = P.BroadcastTo(shape)
        >>> broadcast_to(input_x)
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    """

    @prim_attr_register
    def __init__(self, shape):
        """Initialize BroadcastTo"""
        validator.check_value_type("shape", shape, (tuple), self.name)
        validator.check("shape length", len(shape), "", 0, Rel.GT, self.name)
        for i in shape:
            validator.check_positive_int(i, "shape element", self.name)
        self.shape = shape

    def infer_shape(self, x_shape):
        validator.check("input_x shape length", len(x_shape), "target shape", len(self.shape), Rel.LE, self.name)

        reversed_x_shape = tuple(reversed(x_shape))
        reversed_target = tuple(reversed(self.shape))
        for i, v in enumerate(reversed_x_shape):
            if v not in (reversed_target[i], 1):
                raise ValueError(f"Not supported shapes for broadcast, "
                                 f"x_shape: {tuple(x_shape)}, target shape {self.shape}.")
        return self.shape

    def infer_dtype(self, x_dtype):
        validator.check_subclass("input_x", x_dtype, mstype.tensor, self.name)
        return x_dtype


class Meshgrid(PrimitiveWithInfer):
    """
    Generates coordinate matrices from given coordinate tensors.

    Given N one-dimensional coordinate tensors, returns a list outputs of N N-D
    coordinate tensors for evaluating expressions on an N-D grid.


    Args:
        indexing (str): Either 'xy' or 'ij'. Default: 'xy'.
        When the indexing argument is set to 'xy' (the default),
        the broadcasting instructions for the first two dimensions are swapped.

    Inputs:
        - **input_x** (Union[tuple, list]) - A Tuple or list of N 1-D Tensor objects.
          The length of input_x should be greater than 1

    Outputs:
        Tensors, A Tuple of N N-D Tensor objects.

    Examples:
        >>> x = np.array([1, 2, 3, 4]).astype(np.int32)
        >>> y = np.array([5, 6, 7]).astype(np.int32)
        >>> z = np.array([8, 9, 0, 1, 2]).astype(np.int32)
        >>> inputs = (x, y, z)
        >>> meshgrid = P.Meshgrid(indexing="xy")
        >>> meshgrid(inputs)
        (Tensor(shape=[3, 4, 6], dtype=UInt32, value=
         [[[1, 1, 1, 1, 1],
          [2, 2, 2, 2, 2],
          [3, 3, 3, 3, 3],
          [4, 4, 4, 4, 4]],
         [[1, 1, 1, 1, 1],
          [2, 2, 2, 2, 2],
          [3, 3, 3, 3, 3],
          [4, 4, 4, 4, 4]],
         [[1, 1, 1, 1, 1],
          [2, 2, 2, 2, 2],
          [3, 3, 3, 3, 3],
          [4, 4, 4, 4, 4]]]),
         Tensor(shape=[3, 4, 6], dtype=UInt32, value=
         [[[5, 5, 5, 5, 5],
          [5, 5, 5, 5, 5],
          [5, 5, 5, 5, 5],
          [5, 5, 5, 5, 5]],
         [[6, 6, 6, 6, 6],
          [6, 6, 6, 6, 6],
          [6, 6, 6, 6, 6],
          [6, 6, 6, 6, 6]],
         [[7, 7, 7, 7, 7],
          [7, 7, 7, 7, 7],
          [7, 7, 7, 7, 7],
          [7, 7, 7, 7, 7]]]),
         Tensor(shape=[3, 4, 6], dtype=UInt32, value=
         [[[8, 9, 0, 1, 2],
          [8, 9, 0, 1, 2],
          [8, 9, 0, 1, 2],
          [8, 9, 0, 1, 2]],
         [[8, 9, 0, 1, 2],
          [8, 9, 0, 1, 2],
          [8, 9, 0, 1, 2],
          [8, 9, 0, 1, 2]],
         [[8, 9, 0, 1, 2],
          [8, 9, 0, 1, 2],
          [8, 9, 0, 1, 2],
          [8, 9, 0, 1, 2]]]))
    """

    @prim_attr_register
    def __init__(self, indexing="xy"):
        """Init Meshgrid"""
        validator.check_value_type("indexing", indexing, (str), self.name)
        if indexing not in ("xy", "ij"):
            raise ValueError("indexing parameter must be either 'xy' or 'ij'")
        self.indexing = indexing

    def infer_shape(self, x_shape):
        validator.check_value_type("shape", x_shape, [tuple, list], self.name)
        validator.check_int(len(x_shape), 2, Rel.GE, "len of input_x", self.name)
        n = len(x_shape)
        shape_0 = []
        for s in x_shape:
            validator.check_int(len(s), 1, Rel.EQ, 'each_input_rank', self.name)
            shape_0.append(s[0])
        if self.indexing == "xy":
            shape_0[0], shape_0[1] = shape_0[1], shape_0[0]
        out_shape = tuple(tuple(shape_0) for _ in range(n))
        return out_shape

    def infer_dtype(self, x_type):
        validator.check_subclass("input_x[0]", x_type[0], mstype.tensor, self.name)
        n = len(x_type)
        for i in range(1, n):
            validator.check('x_type[%d]' % i, x_type[i], 'base', x_type[0], Rel.EQ, self.name, TypeError)
        return x_type


class InplaceUpdate(PrimitiveWithInfer):
    r"""
    Updates specified rows with values in `v`.

    Args:
        indices (Union[int, tuple]): Indices into the left-most dimension of `x`, and determines which rows of x
            to update with v. It is a int or tuple, whose value is in [0, the first dimension size of x).

    Inputs:
        - **x** (Tensor) - A tensor which to be inplace updated. It can be one of the following data types:
          float32, float16 and int32.
        - **v** (Tensor) - A tensor with the same type as `x` and the same dimension size as `x` except
          the first dimension, which must be the same as the size of `indices`.

    Outputs:
        Tensor, with the same type and shape as the input `x`.

    Examples:
        >>> indices = (0, 1)
        >>> x = Tensor(np.array([[1, 2], [3, 4], [5, 6]]), mindspore.float32)
        >>> v = Tensor(np.array([[0.5, 1.0], [1.0, 1.5]]), mindspore.float32)
        >>> inplace_update = P.InplaceUpdate(indices)
        >>> result = inplace_update(x, v)
        >>> print(result)
        [[0.5, 1.0],
         [1.0, 1.5],
         [5.0, 6.0]]
    """

    @prim_attr_register
    def __init__(self, indices):
        """Initialize InplaceUpdate"""
        self.init_prim_io_names(inputs=['x', 'v'], outputs=['y'])
        self.indices = indices
        validator.check_value_type("indices", indices, [int, tuple], self.name)
        if isinstance(indices, int):
            self.indices = (indices,)
        for item in self.indices:
            validator.check_value_type("item of indices", item, [int], self.name)

    def infer_dtype(self, x_dtype, v_dtype):
        args = {'x': x_dtype, 'v': v_dtype}
        valid_type = [mstype.int32, mstype.float16, mstype.float32]
        validator.check_tensors_dtypes_same_and_valid(args, valid_type, self.name)
        return x_dtype

    def infer_shape(self, x_shape, v_shape):
        validator.check("x", len(x_shape), "v", len(v_shape), Rel.EQ, self.name)
        validator.check("size of indices", len(self.indices), "v's first dimension", v_shape[0],
                        Rel.EQ, self.name)
        for i in self.indices:
            if i < 0 or i >= x_shape[0]:
                raise ValueError(f'The value of indices must be in [0, {x_shape[0]}), but got {i}.')
        x_rank = len(x_shape)
        for idx in range(x_rank)[1:]:
            validator.check('v dim %d' % idx, v_shape[idx], "x dim %d" % idx, x_shape[idx], Rel.EQ, self.name)
        return x_shape


class ReverseSequence(PrimitiveWithInfer):
    """
    Reverses variable length slices.

    Args:
        seq_dim (int): The dimension where reversal is performed. Required.
        batch_dim (int): The input is sliced in this dimension. Default: 0.

    Inputs:
        - **x** (Tensor) - The input to reverse, supporting all number types including bool.
        - **seq_lengths** (Tensor) - Must be a 1-D vector with int32 or int64 types.

    Outputs:
        Reversed tensor with the same shape and data type as input.

    Examples:
        >>> x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), mindspore.float32)
        >>> seq_lengths = Tensor(np.array([1, 2, 3]))
        >>> reverse_sequence = P.ReverseSequence(seq_dim=1)
        >>> output = reverse_sequence(x, seq_lengths)
        >>> print(output)
        [[1 2 3]
         [5 4 6]
         [9 8 7]]
    """

    @prim_attr_register
    def __init__(self, seq_dim, batch_dim=0):
        """Initialize ReverseSequence"""
        self.init_prim_io_names(inputs=['x', 'seq_lengths'], outputs=['y'])
        validator.check_value_type("seq_dim", seq_dim, [int], self.name)
        self.seq_dim_ = seq_dim
        validator.check_value_type("batch_dim", batch_dim, [int], self.name)
        self.batch_dim_ = batch_dim

    def infer_shape(self, x, seq_lengths):
        validator.check("seq_dim", self.seq_dim_, "x rank", len(x), Rel.LE, self.name)
        validator.check("batch_dim", self.batch_dim_, "x rank", len(x), Rel.LE, self.name)
        validator.check("batch_dim", self.batch_dim_, "seq_dim", self.seq_dim_, Rel.NE, self.name)
        validator.check("seq_lengths rank", len(seq_lengths), "expected", 1, Rel.EQ, self.name)
        validator.check("seq_lengths vector size", seq_lengths[0],
                        "input size along batch_dim", x[self.batch_dim_], Rel.EQ, self.name)
        return x

    def infer_dtype(self, x, seq_lengths):
        validator.check_tensor_dtype_valid("x_dtype", x, mstype.number_type + (mstype.bool_,), self.name)
        validator.check_tensor_dtype_valid("seq_lengths_dtype", seq_lengths, [mstype.int32, mstype.int64], self.name)
        return x


class EditDistance(PrimitiveWithInfer):
    """
    Computes the Levebshtein Edit Distance. It is used to measure the similarity of two sequences.

    Args:
        normalize (bool): If true, edit distances are normalized by length of truth. Default: True.

    Inputs:
        - **hypothesis_indices** (Tensor) - The indices of the hypothesis list SparseTensor. With int64 data type.
          The shape of tensor is :math:`(N, R)`.
        - **hypothesis_values** (Tensor) - The values of the hypothesis list SparseTensor.
          Must be 1-D vector with length of N.
        - **hypothesis_shape** (Tensor) - The shape of the hypothesis list SparseTensor.
          Must be R-length vector with int64 data type. Only constant value is allowed.
        - **truth_indices** (Tensor) - The indices of the truth list SparseTensor. With int64 data type.
          The shape of tensor is :math:`(M, R)`.
        - **truth_values** (Tensor) - The values of the truth list SparseTensor. Must be 1-D vector with length of M.
        - **truth_shape** (Tensor) - The shape of the truth list SparseTensor.
          Must be R-length vector with int64 data type. Only constant value is allowed.

    Outputs:
        Tensor, a dense tensor with rank `R-1` and float32 data type.

    Examples:
        >>> import numpy as np
        >>> from mindspore import context
        >>> from mindspore import Tensor
        >>> import mindspore.nn as nn
        >>> import mindspore.ops.operations as P
        >>> context.set_context(mode=context.GRAPH_MODE)
        >>> class EditDistance(nn.Cell):
        >>>     def __init__(self, hypothesis_shape, truth_shape, normalize=True):
        >>>         super(EditDistance, self).__init__()
        >>>         self.edit_distance = P.EditDistance(normalize)
        >>>         self.hypothesis_shape = hypothesis_shape
        >>>         self.truth_shape = truth_shape
        >>>
        >>>     def construct(self, hypothesis_indices, hypothesis_values, truth_indices, truth_values):
        >>>         return self.edit_distance(hypothesis_indices, hypothesis_values, self.hypothesis_shape,
        >>>                                   truth_indices, truth_values, self.truth_shape)
        >>>
        >>> hypothesis_indices = Tensor(np.array([[0, 0, 0], [1, 0, 1], [1, 1, 1]]).astype(np.int64))
        >>> hypothesis_values = Tensor(np.array([1, 2, 3]).astype(np.float32))
        >>> hypothesis_shape = Tensor(np.array([1, 1, 2]).astype(np.int64))
        >>> truth_indices = Tensor(np.array([[0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]]).astype(np.int64))
        >>> truth_values = Tensor(np.array([1, 3, 2, 1]).astype(np.float32))
        >>> truth_shape = Tensor(np.array([2, 2, 2]).astype(np.int64))
        >>> edit_distance = EditDistance(hypothesis_shape, truth_shape)
        >>> out = edit_distance(hypothesis_indices, hypothesis_values, truth_indices, truth_values)
        >>> print(out)
        >>> [[1.0, 1.0], [1.0, 1.0]]
    """

    @prim_attr_register
    def __init__(self, normalize=True):
        """Initialize EditDistance"""
        self.normalize = validator.check_value_type("normalize", normalize, [bool], self.name)
        self.set_const_input_indexes([2, 5])

    def __infer__(self, h_indices, h_values, h_shape, truth_indices, truth_values, truth_shape):
        validator.check_const_input('hypothesis_shape', h_shape['value'], self.name)
        validator.check_const_input('truth_shape', truth_shape['value'], self.name)
        args_int = {"hypothesis_indices": h_indices['dtype'], "hypothesis_shape": h_shape['dtype'],
                    "truth_indices": truth_indices['dtype'], "truth_shape": truth_shape['dtype']}
        validator.check_tensors_dtypes_same_and_valid(args_int, [mstype.int64], self.name)
        args = {"hypothesis_values": h_values['dtype'], "truth_values": truth_values['dtype']}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.number_type, self.name)

        hypothesis_indices_shp, truth_indices_shp = h_indices['shape'], truth_indices['shape']
        validator.check("hypothesis_indices rank", len(hypothesis_indices_shp), "expected", 2, Rel.EQ, self.name)
        validator.check("truth_indices rank", len(truth_indices_shp), "expected", 2, Rel.EQ, self.name)
        validator.check("hypothesis_values rank", len(h_values['shape']), "expected", 1, Rel.EQ, self.name)
        validator.check("hypothesis_shape rank", len(h_shape['shape']), "expected", 1, Rel.EQ, self.name)
        validator.check("truth_values rank", len(truth_values['shape']), "expected", 1, Rel.EQ, self.name)
        validator.check("truth_shape rank", len(truth_shape['shape']), "expected", 1, Rel.EQ, self.name)
        validator.check("hypothesis_values shape", h_values['shape'][0],
                        "hypothesis_indices shape[0]", hypothesis_indices_shp[0], Rel.EQ, self.name)
        validator.check("hypothesis_shape", h_shape['shape'][0],
                        "hypothesis_indices shape[1]", hypothesis_indices_shp[1], Rel.EQ, self.name)
        validator.check("truth_values shape", truth_values['shape'][0],
                        "truth_indices shape[0]", truth_indices_shp[0], Rel.EQ, self.name)
        validator.check("hypothesis_shape", h_shape['shape'][0],
                        "truth_shape", truth_shape['shape'][0], Rel.EQ, self.name)
        hypothesis_shape_v = h_shape['value'].asnumpy()
        truth_shape_v = truth_shape['value'].asnumpy()
        out_shape_rank = len(hypothesis_shape_v) - 1
        out_shape = []
        for i in range(out_shape_rank):
            out_shape.append(max(hypothesis_shape_v[i], truth_shape_v[i]))

        return {'shape': tuple(out_shape),
                'dtype': mstype.tensor_type(mstype.float32),
                'value': None}


class TransShape(PrimitiveWithInfer):
    """
    Transforms the shape of input tensor to target shape.

    Inputs:
        - **input_x** (Tensor) - A input tensor.
        - **out_shape** (tuple[int]) - The shape of output data.

    Outputs:
        Tensor, a tensor whose data type is same as 'input_x', and the shape is the same as the `out_shape`.
    """

    @prim_attr_register
    def __init__(self):
        self.__setattr_flag__ = True

    def __infer__(self, x, shape):
        shp = shape['value']
        dtype = x['dtype']
        validator.check_tensor_dtype_valid('x', dtype, mstype.number_type + (mstype.bool_,), self.name)
        self.add_prim_attr('out_shape', tuple(shp))
        return {'shape': shp,
                'dtype': dtype,
                'value': None}


class Sort(PrimitiveWithInfer):
    """
    Sorts the elements of the input tensor along a given dimension in ascending order by value.

    Args:
        axis (int): The dimension to sort along. Default: -1.
        descending (bool): Controls the sorting order. If descending is True then the elements
            are sorted in descending order by value. Default: False.

    Inputs:
        - **x** (Tensor) - The input to sort, with float16 or float32 data type.

    Outputs:
        - **y1** (Tensor) - A tensor whose values are the sorted values, with the same shape and data type as input.
        - **y2** (Tensor) - The indices of the elements in the original input tensor. Data type is int32.

    Examples:
        >>> x = Tensor(np.array([[8, 2, 1], [5, 9, 3], [4, 6, 7]]), mindspore.float16)
        >>> sort = P.Sort()
        >>> sort(x)
        ([[1.0, 2.0, 8.0], [3.0, 5.0, 9.0], [4.0, 6.0 ,7.0]],
         [[2, 1, 0], [2, 0, 1], [0, 1, 2]])
    """

    @prim_attr_register
    def __init__(self, axis=-1, descending=False):
        """Initialize Sort"""
        self.axis = validator.check_value_type("axis", axis, [int], self.name)
        self.descending = validator.check_value_type("descending", descending, [bool], self.name)

    def infer_shape(self, x_shape):
        return x_shape, x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid("x_dtype", x_dtype, [mstype.float32, mstype.float16], self.name)
        return x_dtype, mstype.tensor_type(mstype.int32)


class EmbeddingLookup(PrimitiveWithInfer):
    """
    Returns a slice of input tensor based on the specified indices.

    This Primitive has the similar functionality as GatherV2 operating on `axis = 0`, but has one more inputs:
    `offset`.

    Inputs:
        - **input_params** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
          This represents a Tensor slice, instead of the entire Tensor. Currently, the dimension is restricted to be 2.
        - **input_indices** (Tensor) - The shape of tensor is :math:`(y_1, y_2, ..., y_S)`.
          Specifies the indices of elements of the original Tensor. Values can be out of range of `input_params`,
          and the exceeding part will be filled with 0 in the output.
        - **offset** (int) - Specifies the offset value of this `input_params` slice. Thus the real indices
          are equal to `input_indices` minus `offset`.

    Outputs:
        Tensor, the shape of tensor is :math:`(z_1, z_2, ..., z_N)`.

    Examples:
        >>> input_params = Tensor(np.array([[8, 9], [10, 11], [12, 13], [14, 15]]), mindspore.float32)
        >>> input_indices = Tensor(np.array([[5, 2], [8, 5]]), mindspore.int32)
        >>> offset = 4
        >>> out = P.EmbeddingLookup()(input_params, input_indices, offset)
        >>> print(out)
        [[[10, 11], [0 ,0]], [[0, 0], [10, 11]]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize index_select"""
        self.__setattr_flag__ = True
        self.init_prim_io_names(inputs=['params', 'indices', 'offset'],
                                outputs=['output'])

    def __infer__(self, params, indices, offset):
        validator.check_subclass("params", params['dtype'], mstype.tensor, self.name)
        validator.check_tensor_dtype_valid("indices", indices['dtype'], mstype.int_type, self.name)
        validator.check_subclass("offset", offset['dtype'], mstype.int_, self.name)
        params_shp = params['shape']
        out_shape = indices['shape'] + params_shp[1:]
        if 'max_shape' in indices:
            out_max_shape = indices['max_shape'] + params_shp[1:]
        else:
            out_max_shape = out_shape
        if 'min_shape' in indices:
            out_min_shape = indices['min_shape'] + params_shp[1:]
        else:
            out_min_shape = out_shape
        out = {'shape': out_shape,
               'dtype': params['dtype'],
               'value': None,
               'max_shape': out_max_shape,
               'min_shape': out_min_shape}
        return out


class GatherD(PrimitiveWithInfer):
    """
    Gathers values along an axis specified by dim.

    Inputs:
        - **x** (Tensor) - The source tensor.
        - **dim** (int) - The axis along which to index. It must be int32. Only constant value is allowed.
        - **index** (Tensor) - The indices of elements to gather. It can be one of the following data types:
          int32, int64.

    Outputs:
        Tensor, the shape of tensor is :math:`(z_1, z_2, ..., z_N)`.

    Examples:
        >>> x = Tensor(np.array([[1, 2], [3, 4]]), mindspore.int32)
        >>> index = Tensor(np.array([[0, 0], [1, 0]]), mindspore.int32)
        >>> dim = 1
        >>> out = P.GatherD()(x, dim, index)
        >>> print(out)
        [[1, 1], [4, 3]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize GatherD"""
        self.init_prim_io_names(inputs=['x', 'dim', 'index'], outputs=['output'])

    def __infer__(self, x, dim, index):
        validator.check_subclass("x", x['dtype'], mstype.tensor, self.name)
        validator.check_tensor_dtype_valid("index", index['dtype'], [mstype.int32, mstype.int64], self.name)
        validator.check_subclass("dim", dim['dtype'], [mstype.int32, mstype.int64], self.name)
        x_shp = x['shape']
        idx_shp = index['shape']
        x_rank = len(x_shp)
        idx_rank = len(idx_shp)
        validator.check("x_rank, idx_rank", x_rank, "expected", idx_rank, Rel.EQ, self.name)
        dim_v = dim['value']
        validator.check("dim value", dim_v, "expected", 0, Rel.GE, self.name)
        validator.check("dim value", dim_v, "expected", x_rank, Rel.LT, self.name)
        for i in range(x_rank):
            if i == dim_v:
                continue
            validator.check("x_shp[{0}], idx_shp[{0}]".format(i), x_shp[i], "expected", idx_shp[i], Rel.EQ, self.name)

        out = {'shape': index['shape'],
               'dtype': x['dtype'],
               'value': None}
        return out


class Identity(PrimitiveWithInfer):
    """
    Returns a Tensor with the same shape and contents as input.

    Inputs:
        - **x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        Tensor, the shape of tensor is the same as `input_x`, :math:`(x_1, x_2, ..., x_R)`.

    Examples:
        >>> x = Tensor(np.array([1, 2, 3, 4]), mindspore.int64)
        >>> y = P.Identity()(x)
        >>> print(y)
        [1, 2, 3, 4]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize identity"""

    def __infer__(self, x):
        out = {'shape': x['shape'],
               'dtype': x['dtype'],
               'value': None}
        return out


class RepeatElements(PrimitiveWithInfer):
    """
    Repeat elements of a tensor along an axis, like np.repeat.

    Args:
        rep (int): The number of times to repeat, must be positive, required.
        axis (int): The axis along which to repeat, default 0.

    Inputs:
        - **x** (Tensor) - The tensor to repeat values for. Must be of type int32 or float16.

    Outputs:
        One tensor with values repeated along the specified axis. If x has shape
        (s1, s2, ..., sn) and axis is i, the output will have shape (s1, s2, ..., si * rep, ..., sn)


    Examples:
        >>> x = Tensor(np.array([[0, 1, 2], [3, 4, 5]]), mindspore.int32)
        >>> repeat_elements = P.RepeatElements(rep = 2, axis = 0)
        >>> output = repeat_elements(x)
        >>> print(output)
        [[0, 1, 2],
         [0, 1, 2],
         [3, 4, 5],
         [3, 4, 5]],
    """

    @prim_attr_register
    def __init__(self, rep, axis=0):
        self.init_prim_io_names(inputs=["x"], outputs=["output"])

        validator.check_value_type("rep", rep, [int], self.name)
        self.rep = rep

        validator.check_value_type("axis", axis, [int], self.name)
        self.axis = axis

    def infer_shape(self, x_shape):
        validator.check("rep", self.rep, "", 0, Rel.GT, self.name)
        validator.check("axis", self.axis, "dimension of x", len(x_shape), Rel.LT, self.name)
        validator.check("axis", self.axis, "negative dimension of x", -len(x_shape), Rel.GE, self.name)

        x_shape[self.axis] *= self.rep
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_subclass("x_dtype", x_dtype, mstype.tensor, self.name)
        return x_dtype

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

"""Generate bprop for comm ops"""
import mindspore.common.dtype as mstype
from mindspore.ops import functional as F
from .. import operations as P
from ...common.tensor import RowTensor
from ..composite.multitype_ops.zeros_like_impl import zeros_like
from ..operations.comm_ops import (AllGather, _HostAllGather, AllReduce, _AlltoAll, Broadcast,
                                   _GetTensorSlice, _MirrorOperator, ReduceOp,
                                   ReduceScatter, _HostReduceScatter, _VirtualDiv)
from .grad_base import bprop_getters


@bprop_getters.register(AllReduce)
def get_bprop_all_reduce(self):
    """Generate bprop for AllReduce, do allreduce or allgather, allgather for sparse feature."""

    all_reduce_grad = AllReduce(ReduceOp.SUM, self.group)
    all_gather = AllGather(group=self.group)
    if self.instance_name:
        instance_name = "grad" + self.instance_name
        all_reduce_grad.set_prim_instance_name(instance_name)
    equal = P.Equal()
    cast = P.Cast()
    mul = P.Mul()
    dtype = P.DType()

    if self.op == ReduceOp.PROD:
        raise RuntimeError("The bprop of ReduceOp.PROD is not supported yet.")
    if self.op == ReduceOp.SUM:

        def bprop(x, out, dout):
            if F.issubclass_(F.typeof(dout), mstype.tensor):
                dx = all_reduce_grad(dout)
            else:
                indices = all_gather(dout.indices)
                grad = all_gather(dout.values)
                dx = RowTensor(indices, grad, dout.dense_shape)
            return (dx,)
    else:

        def bprop(x, out, dout):
            if F.issubclass_(F.typeof(dout), mstype.tensor):
                dx = all_reduce_grad(dout)
                z = equal(x, out)
                z = cast(z, dtype(dx))
                dx = mul(dx, z)
            else:
                indices = all_gather(dout.indices)
                grad = all_gather(dout.values)
                z = equal(x, out)
                z = cast(z, dtype(grad))
                grad = mul(grad, z)
                dx = RowTensor(indices, grad, dout.dense_shape)
            return (dx,)
    return bprop


@bprop_getters.register(Broadcast)
def get_bprop_broad_cast(self):
    """Generate bprop for Broadcast."""

    def bprop(x, out, dout):
        return (dout,)
    return bprop


@bprop_getters.register(AllGather)
def get_bprop_all_gather(self):
    """Generate bprop for AllGather"""
    all_gather_grad = ReduceScatter(ReduceOp.SUM, self.group)
    fusion = self.get_attr_dict()["fusion"]
    all_gather_grad.add_prim_attr("fusion", fusion)
    if self.instance_name:
        instance_name = "grad_" + self.instance_name
        all_gather_grad.set_prim_instance_name(instance_name)

    def bprop(x, out, dout):
        dx = all_gather_grad(dout)
        return (dx,)

    return bprop


@bprop_getters.register(_HostAllGather)
def get_bprop_host_all_gather(self):
    """Generate bprop for _HostAllGather"""
    host_all_gather_grad = _HostReduceScatter(ReduceOp.SUM, self.group)
    if self.instance_name:
        instance_name = "grad" + self.instance_name
        host_all_gather_grad.set_prim_instance_name(instance_name)

    def bprop(x, out, dout):
        dx = host_all_gather_grad(dout)
        return (dx,)

    return bprop


@bprop_getters.register(ReduceScatter)
def get_bprop_reduce_scatter(self):
    """Generate bprop for ReduceScatter"""
    reduce_scatter_grad = AllGather(self.group)
    if self.instance_name:
        instance_name = "grad" + self.instance_name
        reduce_scatter_grad.set_prim_instance_name(instance_name)

    if self.op != ReduceOp.SUM:
        raise RuntimeError("The reducescatter bprop only support ReduceOp.SUM until now.")

    def bprop(x, out, dout):
        dx = reduce_scatter_grad(dout)
        return (dx,)

    return bprop


@bprop_getters.register(_HostReduceScatter)
def get_bprop_host_reduce_scatter(self):
    """Generate bprop for _HostReduceScatter"""
    host_reduce_scatter_grad = _HostAllGather(self.group)
    if self.instance_name:
        instance_name = "grad" + self.instance_name
        host_reduce_scatter_grad.set_prim_instance_name(instance_name)

    if self.op != ReduceOp.SUM:
        raise RuntimeError("The hostreducescatter bprop only support ReduceOp.SUM until now.")

    def bprop(x, out, dout):
        dx = host_reduce_scatter_grad(dout)
        return (dx,)

    return bprop


@bprop_getters.register(_AlltoAll)
def get_bprop_all_to_all(self):
    """Generate bprop for AlltoAll."""
    all_to_all_grad = _AlltoAll(self.split_count, self.concat_dim, self.split_dim, self.group)
    if self.instance_name:
        instance_name = "grad" + self.instance_name
        all_to_all_grad.set_prim_instance_name(instance_name)

    def bprop(x, out, dout):
        dx = all_to_all_grad(dout)
        return (dx,)

    return bprop


@bprop_getters.register(_MirrorOperator)
def get_bprop_mirror_operator(self):
    """
    Backpropagator for _MirrorOperator, do allreduce or allgather for the devices in group(only for one group),
    allgather for sparse feature.
    """
    group = self.group
    dev_num = self.dev_num
    mean_flag = self.mean_flag

    all_reduce = AllReduce(group=group)
    all_gather = AllGather(group=group)
    mul = P.Mul()
    cast = P.Cast()

    fusion = 1
    if hasattr(self, 'fusion'):
        fusion = self.fusion
    all_reduce.add_prim_attr("fusion", fusion)
    if hasattr(self, 'parameter'):
        parameter = self.parameter
        all_reduce.add_prim_attr("parameter", parameter)

    if self.instance_name:
        instance_name = "grad_mirror" + self.instance_name
        all_reduce.set_prim_instance_name(instance_name)

    def bprop(x, out, dout):
        if mean_flag:
            if F.issubclass_(F.typeof(dout), mstype.tensor):
                dx = all_reduce(dout)
                float_one = F.scalar_cast(1.0, F.dtype(dx))
                num = F.scalar_cast(dev_num, F.dtype(dx))
                dx = mul(dx, cast(F.scalar_to_array(float_one/num), F.dtype(dx)))
            else:
                indices = all_gather(dout.indices)
                grad = all_gather(dout.values)
                float_one = F.scalar_cast(1.0, F.dtype(grad))
                num = F.scalar_cast(dev_num, F.dtype(grad))
                grad = mul(grad, cast(F.scalar_to_array(float_one/num), F.dtype(grad)))
                dx = RowTensor(indices, grad, dout.dense_shape)
        else:
            if F.issubclass_(F.typeof(dout), mstype.tensor):
                dx = all_reduce(dout)
            else:
                indices = all_gather(dout.indices)
                grad = all_gather(dout.values)
                dx = RowTensor(indices, grad, dout.dense_shape)

        return (dx,)
    return bprop


@bprop_getters.register(_VirtualDiv)
def get_bprop_virtual_div_operator(self):
    """Backpropagator for _VirtualDiv, do Div for the divisor."""
    divisor = self.divisor
    op = P.RealDiv()
    cast = P.Cast()
    dtype = P.DType()

    def bprop(x, out, dout):
        if F.issubclass_(F.typeof(dout), mstype.tensor):
            if F.issubclass_(F.dtype(dout), mstype.bool_) or F.issubclass_(F.dtype(dout), mstype.int32) \
                                     or F.issubclass_(F.dtype(dout), mstype.int16):
                return (dout,)
            dx = op(dout, cast(F.scalar_to_array(divisor), dtype(dout)))
            return (dx,)

        if F.issubclass_(F.typeof(dout), mstype.tuple_):
            dx = ()
            input_nums = F.tuple_len(dout)
            for i in range(input_nums):
                ele_grad = op(dout[i], cast(F.scalar_to_array(divisor), dtype(dout[i])))
                dx = dx + (ele_grad,)
            return (dx,)

        dx = []
        input_nums = F.list_len(dout)
        for i in range(input_nums):
            ele_grad = op(dout[i], cast(F.scalar_to_array(divisor), dtype(dout[i])))
            dx.append(ele_grad)
        return (dx,)
    return bprop


@bprop_getters.register(_GetTensorSlice)
def get_bprop_get_tensor_slice_operator(self):
    """Backpropagator for _GetTensorSlice"""

    def bprop(x, dev_mat, tensor_map, out, dout):
        return (zeros_like(x),)
    return bprop

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
"""grad reducer cell for distributed training"""
from mindspore import context
from mindspore.nn.cell import Cell
from mindspore.communication.management import GlobalComm, get_group_size
from mindspore.common.tensor import RowTensor
from mindspore.ops import functional as F, composite as C, operations as P
from mindspore.ops.operations.comm_ops import AllReduce, AllGather
from mindspore.parallel._auto_parallel_context import auto_parallel_context
import mindspore.common.dtype as mstype

reduce_opt = C.MultitypeFuncGraph("reduce_opt")


def _init_allreduce_operators(length, split_indices):
    """ initialize allreduce communication operators"""
    group = 1
    fusion = ()
    for i in range(length):
        fusion = fusion + (group,)
        if split_indices[group - 1] <= i + 1:
            if group >= len(split_indices):
                continue
            group = group + 1
    index = tuple(range(1, length + 1))
    op_list = ()
    for i in range(length):
        op = AllReduce('sum', GlobalComm.WORLD_COMM_GROUP)
        op.add_prim_attr('fusion', fusion[i])
        op.add_prim_attr('index', index[i])
        op_list = op_list + (op,)
    return op_list


@reduce_opt.register("Number", "Bool", "Function", "Function", "Bool", "Tensor")
def _tensors_allreduce(degree, mean, allgather, allreduce, allreduce_filter, grad):
    """
    Apply allreduce on gradient.

    Args:
        degree (int): The mean coefficient.
        mean (bool): When mean is true, the mean coefficient (degree) would apply on gradients.
        allgather (Primitive): The communication operator for sparse gradients.
        allreduce (Primitive): The communication operator for gradients.
        allreduce_filter (bool): When it is true, allreduce would apply.
        grad (Tensor): The gradient tensor before operation.

    Returns:
        Tensor, the gradient tensor after operation.
    """
    if allreduce_filter:
        grad = allreduce(grad)
        if mean:
            degree = F.scalar_cast(degree, F.dtype(grad))
            cast_op = P.Cast()
            mul_op = P.Mul()
            grad = mul_op(grad, cast_op(F.scalar_to_array(1.0 / degree), F.dtype(grad)))
        return grad
    return grad


@reduce_opt.register("Number", "Bool", "Function", "Function", "Bool", "Tensor", "Bool")
def _tensors_allreduce_ps(degree, mean, allgather, allreduce, allreduce_filter, grad, ps_parameter):
    """
    Apply allreduce on gradient.

    Args:
        degree (int): The mean coefficient.
        mean (bool): When mean is true, the mean coefficient (degree) would apply on gradients.
        allgather (Primitive): The communication operator for sparse gradients.
        allreduce (Primitive): The communication operator for gradients.
        allreduce_filter (bool): When it is true, allreduce would apply.
        grad (Tensor): The gradient tensor before operation.
        ps_parameter (bool): Use parameter server or not.

    Returns:
        Tensor, the gradient tensor after operation.
    """
    if ps_parameter:
        return grad

    if allreduce_filter:
        grad = allreduce(grad)
        if mean:
            degree = F.scalar_cast(degree, F.dtype(grad))
            cast_op = P.Cast()
            mul_op = P.Mul()
            grad = mul_op(grad, cast_op(F.scalar_to_array(1.0/degree), F.dtype(grad)))
        return grad
    return grad


@reduce_opt.register("Number", "Bool", "Function", "Function", "Bool", "RowTensor")
def _tensors_allreduce_with_sparse(degree, mean, allgather, allreduce, allreduce_filter, grad):
    """
    Apply allgather on gradient instead of allreduce for sparse feature.
    Allgather is a communication operation used for distributed deep learning.

    Args:
        degree (int): The mean coefficient.
        mean (bool): When mean is true, the mean coefficient (degree) would apply on gradients.
        allgather (Primitive): The communication operator for sparse gradients.
        allreduce (Primitive): The communication operator for gradients.
        allreduce_filter (bool): When it is true, allgather would apply.
        grad (tuple): The indices, gradient tensor and tensor_shape before operation.

    Returns:
        RowTensor, the gradient after operation.
    """
    if allreduce_filter:
        indices = allgather(grad.indices)
        dout = allgather(grad.values)
        if mean:
            degree = F.scalar_cast(degree, F.dtype(grad.values))
            cast_op = P.Cast()
            mul_op = P.Mul()
            dout = mul_op(dout, cast_op(F.scalar_to_array(1.0 / degree), F.dtype(dout)))
        grad = RowTensor(indices, dout, grad.dense_shape)
    return grad


@reduce_opt.register("Number", "Bool", "Function", "Function", "Bool", "RowTensor", "Bool")
def _tensors_allreduce_with_sparse_ps(degree, mean, allgather, allreduce, allreduce_filter, grad, ps_parameter):
    """
    Apply allgather on gradient instead of allreduce for sparse feature.
    Allgather is a communication operation used for distributed deep learning.

    Args:
        degree (int): The mean coefficient.
        mean (bool): When mean is true, the mean coefficient (degree) would apply on gradients.
        allgather (Primitive): The communication operator for sparse gradients.
        allreduce (Primitive): The communication operator for gradients.
        allreduce_filter (bool): When it is true, allgather would apply.
        grad (tuple): The indices, gradient tensor and tensor_shape before operation.
        ps_parameter (bool): Use parameter server or not.

    Returns:
        RowTensor, the gradient after operation.
    """
    if ps_parameter:
        return grad

    if allreduce_filter:
        indices = allgather(grad.indices)
        dout = allgather(grad.values)
        if mean:
            degree = F.scalar_cast(degree, F.dtype(grad.values))
            cast_op = P.Cast()
            mul_op = P.Mul()
            dout = mul_op(dout, cast_op(F.scalar_to_array(1.0 / degree), F.dtype(dout)))
        grad = RowTensor(indices, dout, grad.dense_shape)
    return grad


_get_datatype = C.MultitypeFuncGraph("_get_datatype")


@_get_datatype.register("Tensor")
def _tensors_get_datatype(grad):
    """
    Acquire gradient datatype.

    Args:
        grad (Tensor): The gradient tensor before operation.

    Returns:
        mstype, the datatype of gradient.
    """
    return F.dtype(grad)


@_get_datatype.register("RowTensor")
def _tensors_get_datatype_with_sparse(grad):
    """
    Acquire gradient datatype.

    Args:
        grad (RowTensor): The gradient before operation.

    Returns:
        mstype, the datatype of gradient.
    """
    return F.dtype(grad.values)


_cast_datatype = C.MultitypeFuncGraph("_cast_datatype")


@_cast_datatype.register("TypeType", "Tensor")
def _tensors_cast_datatype(datatype, grad):
    """
    Cast gradient to datatype.

    Args:
        datatype (mstype): the destination datatype of gradient.
        grad (Tensor): The gradient tensor before operation.

    Returns:
        Tensor, the gradient tensor after operation.
    """
    return F.cast(grad, datatype)


@_cast_datatype.register("TypeType", "RowTensor")
def _tensors_cast_datatype_with_sparse(datatype, grad):
    """
    Cast gradient to datatype.

    Args:
        datatype (mstype): the destination datatype of gradient.
        grad (RowTensor): The gradient before operation.

    Returns:
        RowTensor, the gradient after operation.
    """
    dout = F.cast(grad.values, datatype)
    return RowTensor(grad.indices, dout, grad.dense_shape)


class DistributedGradReducer(Cell):
    """
    A distributed optimizer.

    Constructs a gradient reducer Cell, which applies communication and average operations on
    single-process gradient values.

    Args:
        parameters (list): the parameters to be updated.
        mean (bool): When mean is true, the mean coefficient (degree) would apply on gradients. Default: False.
        degree (int): The mean coefficient. Usually it equals to device number. Default: None.

    Raises:
        ValueError: If degree is not a int or less than 0.

    Examples:
        >>> from mindspore.communication import init, get_group_size
        >>> from mindspore.ops import composite as C
        >>> from mindspore.ops import operations as P
        >>> from mindspore.ops import functional as F
        >>> from mindspore import context
        >>> from mindspore.context import ParallelMode
        >>> from mindspore import nn
        >>> from mindspore import ParameterTuple
        >>> from mindspore.parallel._utils import (_get_device_num, _get_gradients_mean,
        >>>                                        _get_parallel_mode)
        >>>
        >>> device_id = int(os.environ["DEVICE_ID"])
        >>> context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=True,
        >>>                     device_id=int(device_id))
        >>> init()
        >>> context.reset_auto_parallel_context()
        >>> context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL)
        >>>
        >>>
        >>> class TrainingWrapper(nn.Cell):
        >>>     def __init__(self, network, optimizer, sens=1.0):
        >>>         super(TrainingWrapper, self).__init__(auto_prefix=False)
        >>>         self.network = network
        >>>         self.network.add_flags(defer_inline=True)
        >>>         self.weights = optimizer.parameters
        >>>         self.optimizer = optimizer
        >>>         self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        >>>         self.sens = sens
        >>>         self.reducer_flag = False
        >>>         self.grad_reducer = None
        >>>         self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        >>>         if self.parallel_mode in [ParallelMode.DATA_PARALLEL,
        >>>                                            ParallelMode.HYBRID_PARALLEL]:
        >>>             self.reducer_flag = True
        >>>         if self.reducer_flag:
        >>>             mean = _get_gradients_mean()
        >>>             degree = _get_device_num()
        >>>             self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)
        >>>
        >>>     def construct(self, *args):
        >>>         weights = self.weights
        >>>         loss = self.network(*args)
        >>>         sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        >>>         grads = self.grad(self.network, weights)(*args, sens)
        >>>         if self.reducer_flag:
        >>>             # apply grad reducer on grads
        >>>             grads = self.grad_reducer(grads)
        >>>         return F.depend(loss, self.optimizer(grads))
        >>>
        >>> network = Net()
        >>> optimizer = nn.Momentum(network.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> train_cell = TrainingWrapper(network, optimizer)
        >>> inputs = Tensor(np.ones([16, 16]).astype(np.float32))
        >>> label = Tensor(np.zeros([16, 16]).astype(np.float32))
        >>> grads = train_cell(inputs, label)
    """

    def __init__(self, parameters, mean=True, degree=None):
        super(DistributedGradReducer, self).__init__(auto_prefix=False)
        self.map_ = C.Map()
        if degree is None:
            self.degree = get_group_size()
        else:
            if not isinstance(degree, int) or degree <= 0:
                raise ValueError("Parameter 'degree' in DistributedGradReducer should large than 0 and be int")
            self.degree = degree
        self.mean = mean
        self.allreduce_filter = tuple(x.layerwise_parallel is False for x in parameters)
        is_parallel_optimizer = context.get_auto_parallel_context("enable_parallel_optimizer")
        split_indices = auto_parallel_context().get_all_reduce_fusion_split_indices()
        if is_parallel_optimizer and split_indices:
            self.split_fusion = True
            self.op_list = _init_allreduce_operators(len(parameters), split_indices)
        else:
            self.split_fusion = False
            self.allreduce = AllReduce().add_prim_attr('fusion', 1)
        self.allgather = AllGather(GlobalComm.WORLD_COMM_GROUP)
        ps_filter = lambda x: x.is_param_ps
        self.ps_parameters = tuple(ps_filter(x) for x in parameters)
        self.enable_parameter_server = any(self.ps_parameters)

    def construct(self, grads):
        """
        Under certain circumstances, the data precision of grads could be mixed with float16 and float32. Thus, the
        result of AllReduce is unreliable. To solve the problem, grads must be cast to float32 before AllReduce,
        and cast back after the operation.

        Args:
            grads (Union[Tensor, tuple[Tensor]]): The gradient tensor or tuple before operation.

        Returns:
            new_grads (Union[Tensor, tuple[Tensor]]), the gradient tensor or tuple after operation.
        """
        datatypes = self.map_(F.partial(_get_datatype), grads)
        grads = self.map_(F.partial(_cast_datatype, mstype.float32), grads)
        if self.split_fusion:
            if self.enable_parameter_server:
                new_grad = self.map_(F.partial(reduce_opt, self.degree, self.mean, self.allgather),
                                     self.op_list, self.allreduce_filter, grads, self.ps_parameters)
            else:
                new_grad = self.map_(F.partial(reduce_opt, self.degree, self.mean, self.allgather),
                                     self.op_list, self.allreduce_filter, grads)
        else:
            if self.enable_parameter_server:
                new_grad = self.map_(F.partial(reduce_opt, self.degree, self.mean, self.allgather,
                                               self.allreduce), self.allreduce_filter, grads, self.ps_parameters)
            else:
                new_grad = self.map_(F.partial(reduce_opt, self.degree, self.mean, self.allgather,
                                               self.allreduce), self.allreduce_filter, grads)
        new_grad = self.map_(F.partial(_cast_datatype), datatypes, new_grad)
        return new_grad

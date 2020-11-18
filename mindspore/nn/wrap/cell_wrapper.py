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
"""Cell_wrapper."""
from mindspore.parallel._utils import (_get_device_num, _get_gradients_mean,
                                       _get_parallel_mode)
from mindspore.context import ParallelMode
from ...common import dtype as mstype
from ...common.parameter import Parameter, ParameterTuple
from ...ops import composite as C
from ...ops import functional as F
from ...ops import operations as P
from ...ops.operations.comm_ops import _VirtualDataset
from ..cell import Cell
from .grad_reducer import DistributedGradReducer


class WithLossCell(Cell):
    r"""
    Cell with loss function.

    Wraps the network with loss function. This Cell accepts data and label as inputs and
    the computed loss will be returned.

    Args:
        backbone (Cell): The target network to wrap.
        loss_fn (Cell): The loss function used to compute loss.

    Inputs:
        - **data** (Tensor) - Tensor of shape :math:`(N, \ldots)`.
        - **label** (Tensor) - Tensor of shape :math:`(N, \ldots)`.

    Outputs:
        Tensor, a scalar tensor with shape :math:`()`.

    Examples:
        >>> net = Net()
        >>> loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
        >>> net_with_criterion = nn.WithLossCell(net, loss_fn)
        >>>
        >>> batch_size = 2
        >>> data = Tensor(np.ones([batch_size, 3, 64, 64]).astype(np.float32) * 0.01)
        >>> label = Tensor(np.ones([batch_size, 1, 1, 1]).astype(np.int32))
        >>>
        >>> net_with_criterion(data, label)
    """

    def __init__(self, backbone, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data, label):
        out = self._backbone(data)
        return self._loss_fn(out, label)

    @property
    def backbone_network(self):
        """
        Returns the backbone network.

        Returns:
            Cell, the backbone network.
        """
        return self._backbone


class WithGradCell(Cell):
    r"""
    Cell that returns the gradients.

    Wraps the network with backward cell to compute gradients. A network with a loss function is necessary
    as argument. If loss function in None, the network must be a wrapper of network and loss function. This
    Cell accepts '*inputs' as inputs and returns gradients for each trainable parameter.

    Note:
        Run in PyNative mode.

    Args:
        network (Cell): The target network to wrap. The network only supports single output.
        loss_fn (Cell): Primitive loss function used to compute gradients. Default: None.
        sens (Union[None, Tensor, Scalar, Tuple ...]): The sensitive for backpropagation, the type and shape
            must be same as the `network` output. If None, we will fill one to a same type shape of
            output value. Default: None.

    Inputs:
        - **(*inputs)** (Tuple(Tensor)) - Tuple of input tensors with shape :math:`(N, \ldots)`.

    Outputs:
        list, a list of Tensors with identical shapes as trainable weights.

    Examples:
        >>> # For a defined network Net without loss function
        >>> net = Net()
        >>> loss_fn = nn.SoftmaxCrossEntropyWithLogits()
        >>> grad_net = nn.WithGradCell(net, loss_fn)
        >>>
        >>> # For a network wrapped with loss function
        >>> net = Net()
        >>> net_with_criterion = nn.WithLossCell(net, loss_fn)
        >>> grad_net = nn.WithGradCell(net_with_criterion)
    """

    def __init__(self, network, loss_fn=None, sens=None):
        super(WithGradCell, self).__init__(auto_prefix=False)
        self.network = network
        self.loss_fn = loss_fn
        self.weights = ParameterTuple(network.trainable_params())
        self.grad = C.GradOperation(get_by_list=True, sens_param=(sens is not None))
        self.sens = sens
        if loss_fn is None:
            self.network_with_loss = network
        else:
            self.network_with_loss = WithLossCell(self.network, self.loss_fn)
        self.network_with_loss.set_train()

    def construct(self, *inputs):
        weights = self.weights
        if self.sens is None:
            grads = self.grad(self.network_with_loss, weights)(*inputs)
        else:
            grads = self.grad(self.network_with_loss, weights)(*inputs, self.sens)
        return grads


class TrainOneStepCell(Cell):
    r"""
    Network training package class.

    Wraps the network with an optimizer. The resulting Cell is trained with input *inputs.
    The backward graph will be created in the construct function to update the parameter. Different
    parallel modes are available for training.

    Args:
        network (Cell): The training network. The network only supports single output.
        optimizer (Cell): Optimizer for updating the weights.
        sens (Number): The scaling number to be filled as the input of backpropagation. Default value is 1.0.

    Inputs:
        - **(*inputs)** (Tuple(Tensor)) - Tuple of input tensors with shape :math:`(N, \ldots)`.

    Outputs:
        Tensor, a scalar Tensor with shape :math:`()`.

    Examples:
        >>> net = Net()
        >>> loss_fn = nn.SoftmaxCrossEntropyWithLogits()
        >>> optim = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> #1) Using the WithLossCell existing provide
        >>> loss_net = nn.WithLossCell(net, loss_fn)
        >>> train_net = nn.TrainOneStepCell(loss_net, optim)
        >>>
        >>> #2) Using user-defined WithLossCell
        >>>class MyWithLossCell(nn.cell):
        >>>    def __init__(self, backbone, loss_fn):
        >>>        super(WithLossCell, self).__init__(auto_prefix=False)
        >>>        self._backbone = backbone
        >>>        self._loss_fn = loss_fn
        >>>
        >>>    def construct(self, x, y, label):
        >>>        out = self._backbone(x, y)
        >>>        return self._loss_fn(out, label)
        >>>
        >>> loss_net = MyWithLossCell(net, loss_fn)
        >>> train_net = nn.TrainOneStepCell(loss_net, optim)
    """
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.network.add_flags(defer_inline=True)
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = F.identity
        self.parallel_mode = _get_parallel_mode()
        if self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(self.weights, mean, degree)

    def construct(self, *inputs):
        weights = self.weights
        loss = self.network(*inputs)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*inputs, sens)
        grads = self.grad_reducer(grads)
        return F.depend(loss, self.optimizer(grads))


class GetNextSingleOp(Cell):
    """
    Cell to run for getting the next operation.

    Args:
        dataset_types (list[:class:`mindspore.dtype`]): The types of dataset.
        dataset_shapes (list[tuple[int]]): The shapes of dataset.
        queue_name (str): Queue name to fetch the data.

    For detailed information, refer to `ops.operations.GetNext`.
    """

    def __init__(self, dataset_types, dataset_shapes, queue_name):
        super(GetNextSingleOp, self).__init__()
        self.get_next = P.GetNext(dataset_types, dataset_shapes, len(dataset_types), queue_name)

    def construct(self):
        return self.get_next()


class _VirtualDatasetCell(Cell):
    """
    Wrap the network with virtual dataset to convert data parallel layout to model parallel layout.

    _VirtualDataset is a virtual Primitive, it does not exist in the final executing graph. Inputs and outpus
    of _VirtualDataset are distributed in data parallel pattern, tensor redistribution Primitives is inserted
    dynamically during the graph compile process.

    Note:
        Only used in semi auto parallel and auto parallel mode.

    Args:
        backbone (Cell): The target network to wrap.

    Examples:
        >>> net = Net()
        >>> net = _VirtualDatasetCell(net)
    """

    def __init__(self, backbone):
        super(_VirtualDatasetCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._virtual_dataset = _VirtualDataset()

    def construct(self, data, label):
        data_, label_ = self._virtual_dataset(data, label)
        return self._backbone(data_, label_)


class VirtualDatasetCellTriple(Cell):
    """
    Wrap the network with virtual dataset to convert data parallel layout to model parallel layout.

    VirtualDatasetCellTriple is a virtual Primitive, it does not exist in the final executing graph. Inputs and outputs
    of VirtualDatasetCellTriple are distributed in data parallel pattern, tensor redistribution Primitives is inserted
    dynamically during the graph compile process.

    Note:
        Only used in semi auto parallel and auto parallel mode. There are three inputs, as contrary to two inputs in
        _VirtualDatasetCell.

    Args:
        backbone (Cell): The target network to wrap.

    Examples:
        >>> net = Net()
        >>> net = VirtualDatasetCellTriple(net)
    """

    def __init__(self, backbone):
        super(VirtualDatasetCellTriple, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._virtual_dataset = _VirtualDataset()

    def construct(self, a, b, c):
        a_, b_, c_ = self._virtual_dataset(a, b, c)
        return self._backbone(a_, b_, c_)


class WithEvalCell(Cell):
    r"""
    Cell that returns loss, output and label for evaluation.

    This Cell accepts a network and loss function as arguments and computes loss for model.
    It returns loss, output and label to calculate the metrics.

    Args:
        network (Cell): The network Cell.
        loss_fn (Cell): The loss Cell.

    Inputs:
        - **data** (Tensor) - Tensor of shape :math:`(N, \ldots)`.
        - **label** (Tensor) - Tensor of shape :math:`(N, \ldots)`.

    Outputs:
        Tuple, containing a scalar loss Tensor, a network output Tensor of shape :math:`(N, \ldots)`
        and a label Tensor of shape :math:`(N, \ldots)`.

    Examples:
        >>> # For a defined network Net without loss function
        >>> net = Net()
        >>> loss_fn = nn.SoftmaxCrossEntropyWithLogits()
        >>> eval_net = nn.WithEvalCell(net, loss_fn)
    """

    def __init__(self, network, loss_fn, add_cast_fp32=False):
        super(WithEvalCell, self).__init__(auto_prefix=False)
        self._network = network
        self._loss_fn = loss_fn
        self.add_cast_fp32 = add_cast_fp32


    def construct(self, data, label):
        outputs = self._network(data)
        if self.add_cast_fp32:
            label = F.mixed_precision_cast(mstype.float32, label)
            outputs = F.cast(outputs, mstype.float32)
        loss = self._loss_fn(outputs, label)
        return loss, outputs, label


class ParameterUpdate(Cell):
    """
    Cell that updates parameters.

    With this Cell, one can manually update `param` with the input `Tensor`.

    Args:
        param (Parameter): The parameter to be updated manually.

    Raises:
        KeyError: If parameter with the specified name does not exist.

    Examples:
        >>> network = Net()
        >>> param = network.parameters_dict()['learning_rate']
        >>> update = nn.ParameterUpdate(param)
        >>> update.phase = "update_param"
        >>> lr = Tensor(0.001, mindspore.float32)
        >>> update(lr)
    """

    def __init__(self, param):
        super(ParameterUpdate, self).__init__(auto_prefix=False)
        if not isinstance(param, Parameter):
            raise TypeError("`param` must be `Parameter`, but got {}".format(param))
        self._param = param

    def construct(self, x):
        F.assign(self._param, x)
        return x

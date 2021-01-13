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
"""GPT training wrapper"""


import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.communication.management import get_group_size
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore.common.parameter import Parameter
from src.utils import ClipByGlobalNorm

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0  # transformer示例中，grad clip的值为5.0，差异是否影响计算
clip_grad = C.MultitypeFuncGraph("clip_grad")

@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """
    Clip gradients.

    Inputs:
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor], clipped gradients.
    """
    if clip_type not in [0, 1]:
        return grad
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = C.clip_by_value(grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
                                   F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad

# grad_scale = C.MultitypeFuncGraph("grad_scale")
# reciprocal = P.Reciprocal()

# @grad_scale.register("Tensor", "Tensor")
# def tensor_grad_scale(scale, grad):
#     return grad * reciprocal(scale)


grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * F.cast(reciprocal(scale), F.dtype(grad))

_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
grad_overflow = P.FloatStatus()

@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    return grad_overflow(grad)

class GPTTrainOneStepWithLossScaleCell(nn.Cell):
    """
    Encapsulation class of GPT network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
    """
    def __init__(self, network, optimizer, scale_update_cell=None, enable_global_norm=False):
        super(GPTTrainOneStepWithLossScaleCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.add_flags(defer_inline=True)
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.enable_global_norm = enable_global_norm
        self.grad = C.GradOperation(get_by_list=True,
                                    sens_param=True)
        self.reducer_flag = False
        self.allreduce = P.AllReduce()
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = F.identity
        self.degree = 1
        if self.reducer_flag:
            self.degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, False, self.degree)
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.cast = P.Cast()
        # Add GPU config codes
        if context.get_context("device_target") == "GPU":
            self.gpu_target = True
            self.float_status = P.FloatStatus()
            self.addn = P.AddN()
            self.reshape = P.Reshape()
        else:
            self.gpu_target = False
            self.alloc_status = P.NPUAllocFloatStatus()
            self.get_status = P.NPUGetFloatStatus()
            self.clear_before_grad = P.NPUClearFloatStatus()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.depend_parameter_use = P.ControlDepend(depend_mode=1)
        self.base = Tensor(1, mstype.float32)
        self.less_equal = P.LessEqual()
        self.hyper_map = C.HyperMap()
        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32),
                                        name="loss_scale")

    @C.add_flags(has_effect=True)
    def construct(self,
                  input_ids,
                  past=None,
                  sens=None):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(input_ids,
                            past)

        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens
        ########
        init = False
        ############
        
        # alloc status and clear should be right before gradoperation
        if not self.gpu_target:
            init = self.alloc_status()
            self.clear_before_grad(init)
        grads = self.grad(self.network, weights)(input_ids,
                                                 past,
                                                 self.cast(scaling_sens,
                                                           mstype.float32))
        # apply grad reducer on grads(这里和transformer示例不一致，reducer_flag=False是不执行grad_reducer的)
        grads = self.grad_reducer(grads)
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens * self.degree), grads)

        if self.enable_global_norm:
            grads = ClipByGlobalNorm()(grads)
        else:
            grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)

        ###
        if not self.gpu_target:
            self.get_status(init)
            # sum overflow buffer elements, 0: not overflow, >0: overflow
            flag_sum = self.reduce_sum(init, (0,))
            
        ###
        else:
            flag_sum = self.hyper_map(F.partial(_grad_overflow), grads)
            flag_sum = self.addn(flag_sum)
            # convert flag_sum to scalar
            flag_sum = self.reshape(flag_sum, (()))
            
        if self.is_distributed:
            # sum overflow flag over devices
            flag_reduce = self.allreduce(flag_sum)
            cond = self.less_equal(self.base, flag_reduce)
        else:
            cond = self.less_equal(self.base, flag_sum)
            
        overflow = cond
        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, cond)
        if overflow:
            succ = False
        else:
            succ = self.optimizer(grads)
            
        ret = (loss, cond, scaling_sens)
        return F.depend(ret, succ)

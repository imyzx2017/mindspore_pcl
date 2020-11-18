# Copyright 2019 Huawei Technologies Co., Ltd
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

"""SqueezeGrad op"""
from mindspore.ops.op_info_register import op_info_register, AkgGpuRegOp, DataType

squeeze_grad_op_info = AkgGpuRegOp("SqueezeGrad") \
    .fusion_type("OPAQUE") \
    .input(0, "y_grad") \
    .output(0, "output") \
    .attr("x_shape", "required", "listInt") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default) \
    .dtype_format(DataType.I16_Default, DataType.I16_Default) \
    .dtype_format(DataType.U8_Default, DataType.U8_Default) \
    .dtype_format(DataType.BOOL_Default, DataType.BOOL_Default) \
    .get_op_info()


@op_info_register(squeeze_grad_op_info)
def _squeeze_grad_akg():
    """SqueezeGrad AutoDiff register"""
    return

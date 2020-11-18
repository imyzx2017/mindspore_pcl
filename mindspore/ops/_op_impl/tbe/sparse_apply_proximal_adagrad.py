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

"""SparseApplyProximalAdagradD op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

sparse_apply_proximal_adagrad_d_op_info = TBERegOp("SparseApplyProximalAdagrad") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("sparse_apply_proximal_adagrad.so") \
    .compute_cost(10) \
    .kernel_name("sparse_apply_proximal_adagrad_d") \
    .partial_flag(True) \
    .attr("use_locking", "optional", "bool", "true,false", "false") \
    .input(0, "var", False, "required", "all") \
    .input(1, "accum", False, "required", "all") \
    .input(2, "lr", False, "required", "all") \
    .input(3, "l1", False, "required", "all") \
    .input(4, "l2", False, "required", "all") \
    .input(5, "grad", False, "required", "all") \
    .input(6, "indices", False, "required", "all") \
    .output(0, "var", False, "required", "all") \
    .output(1, "accum", False, "required", "all") \
    .dtype_format(DataType.F32_NCHW, DataType.F32_NCHW, DataType.F32_NCHW, DataType.F32_NCHW,
                  DataType.F32_NCHW, DataType.F32_NCHW, DataType.I16_NCHW, DataType.F32_NCHW,
                  DataType.F32_NCHW) \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD,
                  DataType.F32_5HD, DataType.F32_5HD, DataType.I16_5HD, DataType.F32_5HD,
                  DataType.F32_5HD) \
    .dtype_format(DataType.F32_NHWC, DataType.F32_NHWC, DataType.F32_NHWC, DataType.F32_NHWC,
                  DataType.F32_NHWC, DataType.F32_NHWC, DataType.I16_NHWC, DataType.F32_NHWC,
                  DataType.F32_NHWC) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.I16_Default, DataType.F32_Default,
                  DataType.F32_Default) \
    .dtype_format(DataType.F32_FracZ, DataType.F32_FracZ, DataType.F32_FracZ, DataType.F32_FracZ,
                  DataType.F32_FracZ, DataType.F32_FracZ, DataType.I16_FracZ, DataType.F32_FracZ,
                  DataType.F32_FracZ) \
    .dtype_format(DataType.F32_NCHW, DataType.F32_NCHW, DataType.F32_NCHW, DataType.F32_NCHW,
                  DataType.F32_NCHW, DataType.F32_NCHW, DataType.I32_NCHW, DataType.F32_NCHW,
                  DataType.F32_NCHW) \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD,
                  DataType.F32_5HD, DataType.F32_5HD, DataType.I32_5HD, DataType.F32_5HD,
                  DataType.F32_5HD) \
    .dtype_format(DataType.F32_NHWC, DataType.F32_NHWC, DataType.F32_NHWC, DataType.F32_NHWC,
                  DataType.F32_NHWC, DataType.F32_NHWC, DataType.I32_NHWC, DataType.F32_NHWC,
                  DataType.F32_NHWC) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.I32_Default, DataType.F32_Default,
                  DataType.F32_Default) \
    .dtype_format(DataType.F32_FracZ, DataType.F32_FracZ, DataType.F32_FracZ, DataType.F32_FracZ,
                  DataType.F32_FracZ, DataType.F32_FracZ, DataType.I32_FracZ, DataType.F32_FracZ,
                  DataType.F32_FracZ) \
    .dtype_format(DataType.F32_NCHW, DataType.F32_NCHW, DataType.F32_NCHW, DataType.F32_NCHW,
                  DataType.F32_NCHW, DataType.F32_NCHW, DataType.I64_NCHW, DataType.F32_NCHW,
                  DataType.F32_NCHW) \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD,
                  DataType.F32_5HD, DataType.F32_5HD, DataType.I64_5HD, DataType.F32_5HD,
                  DataType.F32_5HD) \
    .dtype_format(DataType.F32_NHWC, DataType.F32_NHWC, DataType.F32_NHWC, DataType.F32_NHWC,
                  DataType.F32_NHWC, DataType.F32_NHWC, DataType.I64_NHWC, DataType.F32_NHWC,
                  DataType.F32_NHWC) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.I64_Default, DataType.F32_Default,
                  DataType.F32_Default) \
    .dtype_format(DataType.F32_FracZ, DataType.F32_FracZ, DataType.F32_FracZ, DataType.F32_FracZ,
                  DataType.F32_FracZ, DataType.F32_FracZ, DataType.I64_FracZ, DataType.F32_FracZ,
                  DataType.F32_FracZ) \
    .dtype_format(DataType.F32_NCHW, DataType.F32_NCHW, DataType.F32_NCHW, DataType.F32_NCHW,
                  DataType.F32_NCHW, DataType.F32_NCHW, DataType.U16_NCHW, DataType.F32_NCHW,
                  DataType.F32_NCHW) \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD,
                  DataType.F32_5HD, DataType.F32_5HD, DataType.U16_5HD, DataType.F32_5HD,
                  DataType.F32_5HD) \
    .dtype_format(DataType.F32_NHWC, DataType.F32_NHWC, DataType.F32_NHWC, DataType.F32_NHWC,
                  DataType.F32_NHWC, DataType.F32_NHWC, DataType.U16_NHWC, DataType.F32_NHWC,
                  DataType.F32_NHWC) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.U16_Default, DataType.F32_Default,
                  DataType.F32_Default) \
    .dtype_format(DataType.F32_FracZ, DataType.F32_FracZ, DataType.F32_FracZ, DataType.F32_FracZ,
                  DataType.F32_FracZ, DataType.F32_FracZ, DataType.U16_FracZ, DataType.F32_FracZ,
                  DataType.F32_FracZ) \
    .dtype_format(DataType.F32_NCHW, DataType.F32_NCHW, DataType.F32_NCHW, DataType.F32_NCHW,
                  DataType.F32_NCHW, DataType.F32_NCHW, DataType.U32_NCHW, DataType.F32_NCHW,
                  DataType.F32_NCHW) \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD,
                  DataType.F32_5HD, DataType.F32_5HD, DataType.U32_5HD, DataType.F32_5HD,
                  DataType.F32_5HD) \
    .dtype_format(DataType.F32_NHWC, DataType.F32_NHWC, DataType.F32_NHWC, DataType.F32_NHWC,
                  DataType.F32_NHWC, DataType.F32_NHWC, DataType.U32_NHWC, DataType.F32_NHWC,
                  DataType.F32_NHWC) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.U32_Default, DataType.F32_Default,
                  DataType.F32_Default) \
    .dtype_format(DataType.F32_FracZ, DataType.F32_FracZ, DataType.F32_FracZ, DataType.F32_FracZ,
                  DataType.F32_FracZ, DataType.F32_FracZ, DataType.U32_FracZ, DataType.F32_FracZ,
                  DataType.F32_FracZ) \
    .dtype_format(DataType.F32_NCHW, DataType.F32_NCHW, DataType.F32_NCHW, DataType.F32_NCHW,
                  DataType.F32_NCHW, DataType.F32_NCHW, DataType.U64_NCHW, DataType.F32_NCHW,
                  DataType.F32_NCHW) \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD,
                  DataType.F32_5HD, DataType.F32_5HD, DataType.U64_5HD, DataType.F32_5HD,
                  DataType.F32_5HD) \
    .dtype_format(DataType.F32_NHWC, DataType.F32_NHWC, DataType.F32_NHWC, DataType.F32_NHWC,
                  DataType.F32_NHWC, DataType.F32_NHWC, DataType.U64_NHWC, DataType.F32_NHWC,
                  DataType.F32_NHWC) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.U64_Default, DataType.F32_Default,
                  DataType.F32_Default) \
    .dtype_format(DataType.F32_FracZ, DataType.F32_FracZ, DataType.F32_FracZ, DataType.F32_FracZ,
                  DataType.F32_FracZ, DataType.F32_FracZ, DataType.U64_FracZ, DataType.F32_FracZ,
                  DataType.F32_FracZ) \
    .get_op_info()


@op_info_register(sparse_apply_proximal_adagrad_d_op_info)
def _sparse_apply_proximal_adagrad():
    """SparseApplyProximalAdagradD TBE register"""
    return
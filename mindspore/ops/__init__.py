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

"""
Operators can be used in the construct function of Cell.

Examples:

    >>> from mindspore.ops import operations as P
    >>> from mindspore.ops import composite as C
    >>> from mindspore.ops import functional as F
    >>> import mindspore.ops as ops

Note:
    - The Primitive operators in operations need to be used after instantiation.
    - The composite operators are the pre-defined combination of operators.
    - The functional operators are the pre-instantiated Primitive operators, which can be used directly as a function.
    - For functional operators usage, please refer to
      https://gitee.com/mindspore/mindspore/blob/master/mindspore/ops/functional.py
"""

from .primitive import Primitive, PrimitiveWithInfer, prim_attr_register
from .vm_impl_registry import get_vm_impl_fn, vm_impl_registry
from .op_info_register import op_info_register, AkgGpuRegOp, AkgAscendRegOp, AiCPURegOp, TBERegOp, DataType
from .primitive import constexpr
from . import composite, operations, functional
from . import signature
from .composite import *
from .operations import *
from .functional import *

__primitive__ = [
    "prim_attr_register", "Primitive", "PrimitiveWithInfer", "signature"
]

__all__ = ["get_vm_impl_fn", "vm_impl_registry",
           "op_info_register", "AkgGpuRegOp", "AkgAscendRegOp", "AiCPURegOp", "TBERegOp", "DataType",
           "constexpr"]
__all__.extend(__primitive__)
__all__.extend(composite.__all__)
__all__.extend(operations.__all__)
__all__.extend(functional.__all__)

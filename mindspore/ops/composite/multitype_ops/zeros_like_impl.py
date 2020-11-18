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

"""Implementation for internal polymorphism `zeros_like_leaf` operations."""

from ...composite import base
from ... import functional as F


zeros_like_leaf = base.MultitypeFuncGraph('zeros_like_leaf', True)
"""
`zeros_like_leaf` is a metafuncgraph object which will generate a tensor filled with one according to its input type
using ".register" decorator.
"""


@zeros_like_leaf.register("Number")
def _zeros_like_scala(x):
    """Returns 0 which has the same dtype as x where x is a scalar."""
    return 0

@zeros_like_leaf.register("Bool")
def _zeros_like_bool(x):
    """Returns False if x is a bool."""
    return False

newenv = base.EnvInstance_()


@zeros_like_leaf.register("Function")
def _zeros_like_func(x):
    """
    Derivation of a function.

    Args:
        x (Function): x

    Returns:
        EnvInstance_, value is newenv.
    """
    # Unused parameters are placeholders.
    return newenv


@zeros_like_leaf.register("Tensor")
def _zeros_like_tensor(x):
    """Returns a tensor with the same shape and dtype as x and all elements ars 1."""
    return F.zeros_like(x)


@zeros_like_leaf.register("TypeType")
def _zeros_like_type_type(x):
    """Returns x because x is a type. This is usually used in backprop progress."""
    return x


@zeros_like_leaf.register("None")
def _zeros_like_type_none(x):
    """Returns None where x is and should be None. This is usually used in backprop progress."""
    return x


@zeros_like_leaf.register("RefKeyType")
def _zeros_like_refkey_type(x):
    """
    Derivation of a type.

    Args:
        x (RefKeyType): x

    Returns:
        RefKeyType.
    """
    return x


@zeros_like_leaf.register("Problem")
def _zeros_like_abstract_error(x):
    """
    Derivation of a AbstractError.

    Args:
        x (AbstractError): return x

    Returns:
        x.
    """
    return x


# zeros_like is an object that will generate graph of zero_like operation for different type
zeros_like = base.HyperMap(zeros_like_leaf)
"""`zeros_like` is an object that will generate graph of `zero_like` operation for different type."""

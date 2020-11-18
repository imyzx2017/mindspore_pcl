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
"""Utils for mass model."""
from .dictionary import Dictionary
from .ppl_score import ngram_ppl
from .lr_scheduler import square_root_schedule
from .loss_monitor import LossCallBack
from .byte_pair_encoding import bpe_encode
from .initializer import zero_weight, one_weight, normal_weight, weight_variable
from .rouge_score import rouge
from .eval_score import get_score

__all__ = [
    "Dictionary",
    "rouge",
    "bpe_encode",
    "ngram_ppl",
    "square_root_schedule",
    "LossCallBack",
    "one_weight",
    "zero_weight",
    "normal_weight",
    "weight_variable",
    "get_score"
]

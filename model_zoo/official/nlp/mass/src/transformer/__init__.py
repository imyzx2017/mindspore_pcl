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
"""Transformer model module."""
from .transformer import Transformer
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
from .beam_search import BeamSearchDecoder
from .transformer_for_train import TransformerTraining, LabelSmoothedCrossEntropyCriterion, \
    TransformerNetworkWithLoss, TransformerTrainOneStepWithLossScaleCell
from .infer_mass import infer, infer_ppl

__all__ = [
    "infer",
    "infer_ppl",
    "TransformerTraining",
    "LabelSmoothedCrossEntropyCriterion",
    "TransformerTrainOneStepWithLossScaleCell",
    "TransformerNetworkWithLoss",
    "Transformer",
    "TransformerEncoder",
    "TransformerDecoder",
    "BeamSearchDecoder"
]

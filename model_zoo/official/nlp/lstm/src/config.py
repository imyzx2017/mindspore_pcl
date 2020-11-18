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
network config setting
"""
from easydict import EasyDict as edict

# LSTM CONFIG
lstm_cfg = edict({
    'num_classes': 2,
    'learning_rate': 0.1,
    'momentum': 0.9,
    'num_epochs': 20,
    'batch_size': 64,
    'embed_size': 300,
    'num_hiddens': 100,
    'num_layers': 2,
    'bidirectional': True,
    'save_checkpoint_steps': 390,
    'keep_checkpoint_max': 10
})

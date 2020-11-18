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
export checkpoint file into models
"""
import argparse
import numpy as np

import mindspore as ms
from mindspore import Tensor, load_checkpoint, load_param_into_net, export

from src.config import config_gpu as cfg
from src.inception_v3 import InceptionV3

parser = argparse.ArgumentParser(description='inceptionv3 export')
parser.add_argument('--ckpt_file', type=str, required=True, help='inceptionv3 ckpt file.')
parser.add_argument('--output_file', type=str, default='inceptionv3.air', help='inceptionv3 output air name.')
parser.add_argument('--file_format', type=str, choices=["AIR", "ONNX", "MINDIR"], default='AIR', help='file format')
parser.add_argument('--width', type=int, default=299, help='input width')
parser.add_argument('--height', type=int, default=299, help='input height')
args = parser.parse_args()

if __name__ == '__main__':
    net = InceptionV3(num_classes=cfg.num_classes, is_training=False)
    param_dict = load_checkpoint(args.ckpt_file)
    load_param_into_net(net, param_dict)

    input_arr = Tensor(np.random.uniform(0.0, 1.0, size=[cfg.batch_size, 3, args.width, args.height]), ms.float32)

    export(net, input_arr, file_name=args.output_file, file_format=args.file_format)

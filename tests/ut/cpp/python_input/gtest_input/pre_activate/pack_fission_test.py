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

from mindspore.ops import operations as P
from mindspore.ops import Primitive

pack = P.Pack()
concat = P.Concat()
make_tuple = Primitive('make_tuple')


class FnDict:
    def __init__(self):
        self.fnDict = {}

    def __call__(self, fn):
        self.fnDict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fnDict[name]


def test_pack_fission(tag):
    """ test_adam_apply_one_with_decay_rule """
    fns = FnDict()

    @fns
    def before(input0, input1, input2, input3, input4, input5, input6, input7, input8):
        return pack((input0, input1, input2, input3, input4, input5, input6, input7, input8))

    @fns
    def after_divided_by_3(input0, input1, input2, input3, input4, input5, input6, input7, input8):
        pack1 = pack(input0, input1, input2)
        pack2 = pack(input3, input4, input5)
        pack3 = pack(input6, input7, input8)
        return make_tuple(concat(pack1, pack2, pack3))

    @fns
    def after_divided_by_4(input0, input1, input2, input3, input4, input5, input6, input7, input8):
        pack1 = pack(input0, input1, input2, input3)
        pack2 = pack(input4, input5, input6, input7)
        pack3 = pack(input8)
        return make_tuple(concat(pack1, pack2, pack3))

    return fns[tag]

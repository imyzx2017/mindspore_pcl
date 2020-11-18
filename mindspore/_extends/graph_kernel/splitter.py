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
"""GraphKernel splitter"""

import json
import json.decoder as jd
import traceback
from mindspore import log as logger
from . import model


def split_with_json(json_str: str):
    """Call costmodel to split GraphKernel"""
    try:
        graph_desc = json.loads(json_str)
        comp = model.load_composite(graph_desc)
        graph_split, graph_mode = model.split(comp.graph)
        is_multi_graph = len(graph_split) > 1
        graph_list = list(map(comp.dump, graph_split))
        result = {"multi_graph": is_multi_graph,
                  "graph_desc": graph_list,
                  "graph_mode": graph_mode}
        return json.dumps(result)
    except jd.JSONDecodeError:
        logger.error(traceback.format_exc())
        return None

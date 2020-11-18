/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tools/converter/legacy_optimizer/graph/unused_node_remove_pass.h"
#include <queue>
#include "src/common/log_adapter.h"
#include "tools/common/graph_util.h"
#include "include/errorcode.h"
#include "schema/inner/model_generated.h"

namespace mindspore {
namespace lite {
STATUS UnusedNodeRemovePass::Run(schema::MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);
  bool ifChanged = false;
  for (size_t i = 0; i < graph->nodes.size(); i++) {
    auto &node = graph->nodes.at(i);
    if (node->primitive->value.type == schema::PrimitiveType_TupleGetItem) {
      ifChanged = true;
      auto status = IsolateOneWayNode(graph, i);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "IsolateOneWayNode failed, subGraph: " << graph->name << ", node: " << node->name
                      << ", error: " << status;
        return status;
      }
    }
  }
  return ifChanged ? RET_OK : RET_NO_CHANGE;
}
}  // namespace lite
}  // namespace mindspore

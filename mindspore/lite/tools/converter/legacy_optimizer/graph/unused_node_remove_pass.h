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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_LEGACY_OPTIMIZER_GRAGP_UNUSED_NODE_REMOVE_PASS_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_LEGACY_OPTIMIZER_GRAGP_UNUSED_NODE_REMOVE_PASS_H

#include <unordered_map>
#include "tools/converter/optimizer.h"

namespace mindspore {
namespace lite {
class UnusedNodeRemovePass : public GraphPass {
 public:
  UnusedNodeRemovePass() = default;

  ~UnusedNodeRemovePass() override = default;

  STATUS Run(schema::MetaGraphT *graph) override;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_LEGACY_OPTIMIZER_GRAGP_UNUSED_NODE_REMOVE_PASS_H

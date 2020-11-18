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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_FORMAT_TYPE_REMOVE_NO_USE_RESHAPE_OP_H
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_FORMAT_TYPE_REMOVE_NO_USE_RESHAPE_OP_H

#include "backend/optimizer/common/optimizer.h"

namespace mindspore {
namespace opt {
class RemoveNoUseReshapeOp : public PatternProcessPass {
 public:
  explicit RemoveNoUseReshapeOp(bool multigraph = true) : PatternProcessPass("remove_no_use_reshape_op", multigraph) {}
  ~RemoveNoUseReshapeOp() override = default;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_FORMAT_TYPE_REMOVE_NO_USE_RESHAPE_OP_H

/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_COMMON_SUBEXPRESSION_ELIMINATION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_COMMON_SUBEXPRESSION_ELIMINATION_H_
#include "backend/optimizer/common/pass.h"
#include "frontend/optimizer/cse.h"

namespace mindspore {
namespace opt {
class CommonSubexpressionElimination : public Pass {
 public:
  CommonSubexpressionElimination() : Pass("cse") {}
  ~CommonSubexpressionElimination() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;
};

class BackendCSE : public CSE {
 public:
  BackendCSE() = default;
  ~BackendCSE() override = default;
  virtual bool CheckEqualCnodeInputs(const AnfNodePtr &main, const AnfNodePtr &node) const;
  bool CheckReplace(const AnfNodePtr &main, const AnfNodePtr &node, bool check_side_effect = true) const override;
  virtual bool CheckEqualKernelBuildInfo(const AnfNodePtr &main, const AnfNodePtr &node) const;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_COMMON_SUBEXPRESSION_ELIMINATION_H_

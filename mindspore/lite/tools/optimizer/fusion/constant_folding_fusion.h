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

#ifndef MINDSPORE_LITE_SRC_PASS_FUSION_CONSTANT_FOLDING_FUSION_H_
#define MINDSPORE_LITE_SRC_PASS_FUSION_CONSTANT_FOLDING_FUSION_H_

#include "schema/inner/model_generated.h"
#include "src/tensor.h"
#include "src/lite_kernel.h"
#include "nnacl/op_base.h"
#include "backend/optimizer/common/optimizer.h"

namespace mindspore {
namespace opt {
class ConstFoldPass : public PatternProcessPass {
 public:
  explicit ConstFoldPass(bool multigraph = true) : PatternProcessPass("constfold_pass", multigraph) {
    this->context = new lite::InnerContext;
    this->context->Init();
  }
  ~ConstFoldPass() override { delete (this->context); }
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  lite::InnerContext *context = nullptr;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_PASS_FUSION_CONSTANT_FOLDING_FUSION_H_

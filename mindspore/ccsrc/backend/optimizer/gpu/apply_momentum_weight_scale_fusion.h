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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GPU_APPLY_MOMENTUM_WEIGHT_DECAY_SCALE_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GPU_APPLY_MOMENTUM_WEIGHT_DECAY_SCALE_FUSION_H_

#include <memory>
#include "backend/optimizer/common/optimizer.h"

namespace mindspore {
namespace opt {
class ApplyMomentumWeightDecayScaleFusion : public PatternProcessPass {
 public:
  explicit ApplyMomentumWeightDecayScaleFusion(bool multigraph = true)
      : PatternProcessPass("momentum_weightdecay_scale_fusion", multigraph) {
    weight_decay_ = std::make_shared<Var>();
    scale_ = std::make_shared<CondVar>(IsScalar);
    variable_ = std::make_shared<Var>();
    accumulation_ = std::make_shared<Var>();
    learning_rate_ = std::make_shared<Var>();
    gradient_ = std::make_shared<Var>();
    momentum_ = std::make_shared<Var>();
  }
  ~ApplyMomentumWeightDecayScaleFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  static bool IsScalar(const BaseRef &n);

  VarPtr weight_decay_;
  VarPtr scale_;
  VarPtr variable_;
  VarPtr accumulation_;
  VarPtr learning_rate_;
  VarPtr gradient_;
  VarPtr momentum_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GPU_APPLY_MOMENTUM_WEIGHT_DECAY_SCALE_FUSION_H_

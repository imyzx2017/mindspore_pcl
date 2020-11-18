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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_WEIGHT_FORMAT_HARDCODE_PASS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_WEIGHT_FORMAT_HARDCODE_PASS_H_
#include <string>
#include "schema/inner/model_generated.h"
#include "tools/converter/converter_flags.h"
#include "backend/optimizer/common/pass.h"
#include "src/param_value_lite.h"

using mindspore::lite::converter::FmkType;
using mindspore::schema::QuantType;
namespace mindspore::opt {
class WeightFormatHardCodePass : public Pass {
 public:
  WeightFormatHardCodePass() : Pass("weight_format_hardcode_pass") {}
  ~WeightFormatHardCodePass() override = default;
  void SetQuantType(QuantType type);
  void SetFmkType(FmkType fmkType);
  bool Run(const FuncGraphPtr &graph) override;

 private:
  lite::STATUS HardCodeCAFFE(const AnfNodePtr &node, const ParamValueLitePtr &param_value) const;
  lite::STATUS HardCodeONNX(const AnfNodePtr &node, const ParamValueLitePtr &param_value) const;
  lite::STATUS HardCodeMS(const AnfNodePtr &node, const ParamValueLitePtr &param_value) const;
  lite::STATUS HardCodeTFLITE(const AnfNodePtr &node, const ParamValueLitePtr &param_value) const;

 private:
  QuantType quant_type = schema::QuantType_QUANT_NONE;
  FmkType fmk_type = lite::converter::FmkType_TF;
};
}  // namespace mindspore::opt
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_WEIGHT_FORMAT_HARDCODE_PASS_H_

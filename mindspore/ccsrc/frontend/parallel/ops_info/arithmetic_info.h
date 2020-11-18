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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_ARITHMETIC_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_ARITHMETIC_INFO_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "ir/value.h"
#include "frontend/parallel/auto_parallel/operator_costmodel.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
namespace parallel {
class ArithmeticBase : public OperatorInfo {
 public:
  ArithmeticBase(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                 const PrimitiveAttrs &attrs, OperatorCostPtr cost)
      : OperatorInfo(operator_name, inputs_shape, outputs_shape, attrs, cost) {}
  ~ArithmeticBase() override = default;
  Status Init(const StrategyPtr &strategy) override;
  Status InitForCostModel(const StrategyPtr &strategy) override;
  Status GenerateStrategies(int64_t) override;
  Status SetCostUnderStrategy(const StrategyPtr &) override;
  void ReComputeBatchSplitFlagList() override;

 protected:
  Status GetAttrs() override { return SUCCESS; }
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferMirrorOps() override;
  Status InferForwardCommunication() override { return SUCCESS; }
  Status InferTensorInfo() override;
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status InferTensorLayout(TensorLayouts *inputs_layout, TensorLayouts *outputs_layout, const Shape &dev_matrix_array);
  Shapes InferExpendShape();
};

class SubInfo : public ArithmeticBase {
 public:
  SubInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape, const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<ArithmeticCost>(false)) {}
  ~SubInfo() override = default;
};

class TensorAddInfo : public ArithmeticBase {
 public:
  TensorAddInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<ArithmeticCost>(false)) {}
  ~TensorAddInfo() override = default;
};

class MulInfo : public ArithmeticBase {
 public:
  MulInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape, const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<ArithmeticCost>(true)) {}
  ~MulInfo() override = default;
};

class DivInfo : public ArithmeticBase {
 public:
  DivInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape, const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<ArithmeticCost>(true)) {}
  ~DivInfo() override = default;
};

class ModInfo : public ArithmeticBase {
 public:
  ModInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape, const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<ArithmeticCost>(true)) {}
  ~ModInfo() override = default;
};

class RealDivInfo : public ArithmeticBase {
 public:
  RealDivInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
              const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<ArithmeticCost>(true)) {}
  ~RealDivInfo() override = default;
};

class FloorDivInfo : public ArithmeticBase {
 public:
  FloorDivInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
               const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<ArithmeticCost>(true)) {}
  ~FloorDivInfo() override = default;
};

class FloorModInfo : public ArithmeticBase {
 public:
  FloorModInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
               const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<ArithmeticCost>(true)) {}
  ~FloorModInfo() override = default;
};

class PowInfo : public ArithmeticBase {
 public:
  PowInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape, const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<ArithmeticCost>(true)) {}
  ~PowInfo() override = default;
};

class AssignSubInfo : public ArithmeticBase {
 public:
  AssignSubInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<ArithmeticCost>(false)) {}
  ~AssignSubInfo() override = default;
};

class AssignInfo : public ArithmeticBase {
 public:
  AssignInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
             const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<ArithmeticCost>(false)) {}
  ~AssignInfo() override = default;
};

class AssignAddInfo : public ArithmeticBase {
 public:
  AssignAddInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<ArithmeticCost>(false)) {}
  ~AssignAddInfo() override = default;
};

// All dimensions can be split arbitrarily, but the split method of Logits should be the same as that of label.
class SigmoidCrossEntropyWithLogitsInfo : public ArithmeticBase {
 public:
  SigmoidCrossEntropyWithLogitsInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                                    const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<ArithmeticCost>(false)) {}
  ~SigmoidCrossEntropyWithLogitsInfo() override = default;
};

class Atan2Info : public ArithmeticBase {
 public:
  Atan2Info(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
            const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<ArithmeticCost>(false)) {}
  ~Atan2Info() override = default;
};

class DivNoNanInfo : public ArithmeticBase {
 public:
  DivNoNanInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
               const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<ArithmeticCost>(true)) {}
  ~DivNoNanInfo() override = default;
};

class LogicalAndInfo : public ArithmeticBase {
 public:
  LogicalAndInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                 const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<ArithmeticCost>(false)) {}
  ~LogicalAndInfo() override = default;
};

class LogicalOrInfo : public ArithmeticBase {
 public:
  LogicalOrInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<ArithmeticCost>(false)) {}
  ~LogicalOrInfo() override = default;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_ARITHMETIC_INFO_H_

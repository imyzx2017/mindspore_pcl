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

#include "abstract/param_validator.h"

#include <string>
#include <sstream>
#include <memory>
#include "utils/symbolic.h"
#include "abstract/utils.h"

namespace mindspore {
namespace abstract {
#define ABSTRACT_REPORT_NAME_DEC(abstract) constexpr char ReportNameTraits<Abstract##abstract>::name[];

ABSTRACT_REPORT_NAME_DEC(Tensor)
ABSTRACT_REPORT_NAME_DEC(Tuple)
ABSTRACT_REPORT_NAME_DEC(Scalar)
ABSTRACT_REPORT_NAME_DEC(List)
ABSTRACT_REPORT_NAME_DEC(Dictionary)
ABSTRACT_REPORT_NAME_DEC(Slice)
ABSTRACT_REPORT_NAME_DEC(Function)
ABSTRACT_REPORT_NAME_DEC(Type)
ABSTRACT_REPORT_NAME_DEC(KeywordArg)
ABSTRACT_REPORT_NAME_DEC(Class)

TypePtr CheckType(TypePtr type, const TypePtrList &accepts, const std::string &error_message_prefix) {
  bool ok = std::any_of(accepts.begin(), accepts.end(),
                        [type](const TypePtr &accept) -> bool { return IsIdentidityOrSubclass(type, accept); });
  if (ok) {
    return type;
  } else {
    MS_LOG(EXCEPTION) << error_message_prefix << accepts << " but is " << type->ToString();
  }
}

TypePtr CheckTensorDType(const AbstractTensorPtr &tensor, const TypePtrList &accepts,
                         const std::string &error_message_prefix) {
  MS_EXCEPTION_IF_NULL(tensor);
  TypePtr type = tensor->BuildType();
  if (!type->isa<TensorType>()) {
    MS_LOG(EXCEPTION) << error_message_prefix << "requires Tensor but got " << type->ToString();
  }
  TypePtr ele_type = tensor->element()->BuildType();
  if (ele_type == nullptr) {
    MS_LOG(EXCEPTION) << "Abstract tensor element type nullptr";
  }
  return CheckType(ele_type, accepts, error_message_prefix);
}

TypePtr CheckTensorsDTypeSame(const AbstractTensorPtrList &tensor_list, const TypePtrList &accepts,
                              const std::string &error_message_prefix) {
  if (tensor_list.empty()) {
    MS_LOG(EXCEPTION) << "Array list is empty";
  }

  auto sample_tensor = tensor_list[0];
  MS_EXCEPTION_IF_NULL(sample_tensor);
  TypePtr sample_type = sample_tensor->element()->BuildType();
  std::ostringstream loginfoBuffer;
  loginfoBuffer << "same type, got";
  // Check if other elements have the same type with the first element.
  for (size_t index = 1; index < tensor_list.size(); ++index) {
    MS_EXCEPTION_IF_NULL(tensor_list[index]);
    auto aType = tensor_list[index]->element()->BuildType();
    loginfoBuffer << " " << aType->ToString();
    if (sample_type->type_id() != aType->type_id()) {
      MS_LOG(EXCEPTION) << "Expected type " << sample_type->ToString() << ", but got " << aType->ToString()
                        << ", index " << index;
    }
  }
  MS_LOG(DEBUG) << error_message_prefix << loginfoBuffer.str();
  return CheckTensorDType(sample_tensor, accepts, error_message_prefix);
}

TypePtr CheckScalarType(const AbstractScalarPtr &scalar, const TypePtrList &accepts,
                        const std::string &error_message_prefix) {
  if (scalar == nullptr) {
    MS_LOG(EXCEPTION) << "Scalar nullptr";
  }
  auto type = scalar->BuildType();
  if (type == nullptr) {
    MS_LOG(EXCEPTION) << "Scalar value nullptr";
  }

  return CheckType(type, accepts, error_message_prefix);
}

ShapePtr CheckShapeSame(const std::string &op, const AbstractTensorPtr &tensor_base, const AbstractTensorPtr &tensor) {
  ShapePtr shape_base = tensor_base->shape();
  ShapePtr shape = tensor->shape();
  if (*shape != *shape_base) {
    MS_LOG(EXCEPTION) << op << " evaluator first arg shape " << tensor->shape()->ToString()
                      << " are not consistent with second arg shape " << tensor_base->shape()->ToString();
  }
  return shape_base;
}

TypePtr CheckDtypeSame(const std::string &op, const AbstractTensorPtr &tensor_base, const AbstractTensorPtr &tensor) {
  TypePtr type_base = tensor_base->element()->BuildType();
  TypePtr type = tensor->element()->BuildType();
  if (*type != *type_base) {
    MS_LOG(EXCEPTION) << op << " evaluator first arg dtype " << type_base->ToString()
                      << " are not consistent with second arg dtype " << type->ToString();
  }
  return type_base;
}

int64_t CheckAxis(const std::string &op, const ValuePtr &axis, int64_t minimum, int64_t max) {
  if (axis == nullptr) {
    MS_LOG(EXCEPTION) << op << " evaluator axis is null";
  }
  if (!axis->isa<Int64Imm>()) {
    MS_LOG(EXCEPTION) << op << " evaluator axis should be int64_t, but got " << axis->type_name();
  }
  int64_t axis_value = GetValue<int64_t>(axis);
  if (axis_value > max || axis_value < minimum) {
    MS_LOG(EXCEPTION) << op << " evaluator axis value should be in the range [" << minimum << ", " << max
                      << "], but get " << axis_value;
  }
  return axis_value;
}
void CheckArgsSize(const std::string &op, const mindspore::abstract::AbstractBasePtrList &args_spec_list,
                   size_t size_expect) {
  if (args_spec_list.size() != size_expect) {
    MS_LOG(EXCEPTION) << op << " input args size should be " << size_expect << ", but got " << args_spec_list.size();
  }

  for (size_t i = 0; i < size_expect; i++) {
    MS_EXCEPTION_IF_NULL(args_spec_list[i]);
  }
}
}  // namespace abstract
}  // namespace mindspore

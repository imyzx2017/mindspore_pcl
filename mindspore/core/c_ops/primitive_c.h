/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_C_OPS_PRIMITIVE_C_H_
#define MINDSPORE_CORE_C_OPS_PRIMITIVE_C_H_
#include <string>
#include <vector>
#include "ir/primitive.h"
#include "abstract/primitive_infer_map.h"
#include "ir/value.h"
namespace mindspore {
class PrimitiveC : public Primitive {
 public:
  explicit PrimitiveC(const std::string &name) : Primitive(name) {}
  MS_DECLARE_PARENT(PrimitiveC, Primitive);
  ~PrimitiveC() = default;
  AbstractBasePtr Infer(const AbstractBasePtrList &abstract_list);

 protected:
  void InitIOName(const std::vector<std::string> &inputs_name, const std::vector<std::string> &outputs_name);
};
}  // namespace mindspore
#endif  // MINDSPORE_CORE_C_OPS_PRIMITIVE_C_H_

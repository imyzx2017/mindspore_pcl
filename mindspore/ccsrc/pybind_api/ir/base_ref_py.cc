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

#include "pybind_api/ir/base_ref_py.h"

namespace mindspore {
bool PyObjectRef::operator==(const BaseRef &other) const {
  if (!utils::isa<PyObjectRef>(other)) {
    return false;
  }
  return *this == utils::cast<PyObjectRef>(other);
}

bool PyObjectRef::operator==(const PyObjectRef &other) const { return object_.is(other.object_); }
}  // namespace mindspore

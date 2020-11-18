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
#include "src/ops/depend.h"
#include <vector>
#include <memory>

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int Depend::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_Depend;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_Depend) {
    MS_LOG(ERROR) << "primitive_ type is error:" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow)(schema::DependT);
    if (attr == nullptr) {
      MS_LOG(ERROR) << "attr is nullptr";
      return RET_ERROR;
    }
    this->primitive_->value.value = attr;
  }
  return RET_OK;
}
#else
int Depend::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto val_offset = schema::CreateDepend(*fbb);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Depend, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *DependCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Depend>(primitive); }
Registry DependRegistry(schema::PrimitiveType_Depend, DependCreator);
#endif
}  // namespace lite
}  // namespace mindspore

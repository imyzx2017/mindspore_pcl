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

#include "src/ops/while.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"

namespace mindspore {
namespace lite {

typedef struct WhileParemeter {
  OpParameter op_parameter_;
  int body_subgraph_index;
  int cond_subgraph_index;
} WhileParemeter;

OpParameter *PopulateWhileParemeter(const mindspore::lite::PrimitiveC *primitive) {
  WhileParemeter *while_paremeter = reinterpret_cast<WhileParemeter *>(malloc(sizeof(WhileParemeter)));
  if (while_paremeter == nullptr) {
    MS_LOG(ERROR) << "malloc WhileParemeter failed.";
    return nullptr;
  }
  memset(while_paremeter, 0, sizeof(WhileParemeter));
  auto param = reinterpret_cast<mindspore::lite::While *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  while_paremeter->op_parameter_.type_ = primitive->Type();
  while_paremeter->body_subgraph_index = param->GetBodySubgraphIndex();
  while_paremeter->cond_subgraph_index = param->GetCondSubgraphIndex();
  return reinterpret_cast<OpParameter *>(while_paremeter);
}
Registry WhileParemeterRegistry(schema::PrimitiveType_While, PopulateWhileParemeter);
}  // namespace lite
}  // namespace mindspore

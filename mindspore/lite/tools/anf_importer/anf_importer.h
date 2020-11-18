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

#ifndef MINDSPORE_LITE_SRC_ANF_IMPORTER_ANF_IMPORTER_H_
#define MINDSPORE_LITE_SRC_ANF_IMPORTER_ANF_IMPORTER_H_

#include <unordered_map>
#include "ir/func_graph.h"
#include "ir/anf.h"
#include "base/base.h"
#include "schema/inner/model_generated.h"

namespace mindspore::lite {
class AnfImporter {
 public:
  AnfImporter() = default;

  virtual ~AnfImporter() = default;

  virtual int Import(const schema::QuantType &quantType = schema::QuantType_QUANT_NONE);

  virtual FuncGraphPtr GetResult() = 0;

 protected:
  // convert const tensor into parameter and save in nodes_
  virtual int ConverterConstTensor() = 0;
  // convert other node into cnode and save in nodes_
  virtual int ConverterCNode() = 0;

  virtual int AddReturnCNode() = 0;

  AnfNodePtr GetNode(int tensor_id);

  void AddNode(int tensor_id, AnfNodePtr node);

 protected:
  std::unordered_map<int, AnfNodePtr> nodes_;
};
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_ANF_IMPORTER_ANF_IMPORTER_H_

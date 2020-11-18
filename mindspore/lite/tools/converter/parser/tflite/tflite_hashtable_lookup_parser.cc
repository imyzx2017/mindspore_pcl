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

#include "tools/converter/parser/tflite/tflite_hashtable_lookup_parser.h"
#include <vector>
#include <memory>
#include <map>

namespace mindspore {
namespace lite {
STATUS TfliteHashtableLookupParser::Parse(TfliteTensorsInfo *tensors_info,
                                          const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                          const std::unique_ptr<tflite::ModelT> &tflite_model,
                                          const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                          schema::CNodeT *op) {
  MS_LOG(DEBUG) << "parse TfliteHashtableLookupParser";
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  std::unique_ptr<schema::HashtableLookupT> attr = std::make_unique<schema::HashtableLookupT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }

  op->primitive->value.type = schema::PrimitiveType_HashtableLookup;
  op->primitive->value.value = attr.release();
  for (size_t i = 0; i < tflite_op->inputs.size(); ++i) {
    AddOpInput(op, tensors_info, tflite_op->inputs[i], tflite_subgraph->tensors.size(), schema::Format::Format_NHWC);
  }
  for (size_t i = 0; i < tflite_op->outputs.size(); ++i) {
    AddOpOutput(op, tensors_info, tflite_op->outputs[i], tflite_subgraph->tensors.size(), schema::Format::Format_NHWC);
  }
  return RET_OK;
}

TfliteNodeRegister g_tfliteHashtableLookupParser("HashtableLookup", new TfliteHashtableLookupParser());
}  // namespace lite
}  // namespace mindspore

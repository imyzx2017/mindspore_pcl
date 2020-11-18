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

#include "tools/converter/parser/tflite/tflite_scatter_nd_parser.h"
#include <vector>
#include <memory>
#include <utility>
#include <map>

namespace mindspore {
namespace lite {
STATUS TfliteScatterNdParser::Parse(TfliteTensorsInfo *tensors_info,
                                    const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                    const std::unique_ptr<tflite::ModelT> &tflite_model,
                                    const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph, schema::CNodeT *op) {
  MS_LOG(DEBUG) << "parse TfliteScatterNdParser";
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  std::unique_ptr<schema::ScatterNDT> attr = std::make_unique<schema::ScatterNDT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }

  const auto &tflite_attr = tflite_op->builtin_options.AsScatterNdOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op: " << op->name << " attr failed";
    return RET_NULL_PTR;
  }
  op->primitive->value.type = schema::PrimitiveType_ScatterND;
  op->primitive->value.value = attr.release();

  // in tflite, kIndices = 0, kUpdates = 1, kShape = 2
  // in mslite, kScatterShapeIndex = 0, kScatterIndicesIndex = 1, kScatterUpdateIndex = 2;
  AddOpInput(op, tensors_info, tflite_op->inputs[2], tflite_subgraph->tensors.size(), schema::Format::Format_NHWC);
  AddOpInput(op, tensors_info, tflite_op->inputs[0], tflite_subgraph->tensors.size(), schema::Format::Format_NHWC);
  AddOpInput(op, tensors_info, tflite_op->inputs[1], tflite_subgraph->tensors.size(), schema::Format::Format_NHWC);
  AddOpOutput(op, tensors_info, tflite_op->outputs[0], tflite_subgraph->tensors.size(), schema::Format::Format_NHWC);
  return RET_OK;
}

TfliteNodeRegister g_TfliteScatterNdParser("ScatterNd", new TfliteScatterNdParser());
}  // namespace lite
}  // namespace mindspore

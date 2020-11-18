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
 * distributed under the License is distributed on an AS
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tools/converter/parser/tflite/tflite_depth_to_space_parser.h"
#include <vector>
#include <memory>
#include <map>

namespace mindspore {
namespace lite {
STATUS TfliteDepthToSpaceParser::Parse(TfliteTensorsInfo *tensors_info,
                                       const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                       const std::unique_ptr<tflite::ModelT> &tflite_model,
                                       const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph, schema::CNodeT *op) {
  MS_LOG(DEBUG) << "parse TfliteDepthToSpaceParser";

  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  std::unique_ptr<schema::DepthToSpaceT> attr = std::make_unique<schema::DepthToSpaceT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }

  const auto &tflite_attr = tflite_op->builtin_options.AsDepthToSpaceOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op: %s attr failed", op->name.c_str();
    return RET_NULL_PTR;
  }
  attr->blockSize = tflite_attr->block_size;
  attr->format = schema::Format::Format_NHWC;

  op->primitive->value.type = schema::PrimitiveType_DepthToSpace;
  op->primitive->value.value = attr.release();

  AddOpInput(op, tensors_info, tflite_op->inputs[0], tflite_subgraph->tensors.size(), schema::Format::Format_NHWC);
  AddOpOutput(op, tensors_info, tflite_op->outputs[0], tflite_subgraph->tensors.size(), schema::Format::Format_NHWC);
  return RET_OK;
}

TfliteNodeRegister g_tfliteDepthToSpaceParser("DepthToSpace", new TfliteDepthToSpaceParser());
}  // namespace lite
}  // namespace mindspore

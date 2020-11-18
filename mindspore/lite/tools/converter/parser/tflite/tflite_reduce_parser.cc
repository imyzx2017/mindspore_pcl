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

#include "tools/converter/parser/tflite/tflite_reduce_parser.h"
#include <vector>
#include <memory>
#include <string>
#include <map>

namespace mindspore {
namespace lite {
STATUS TfliteReduceParser::Parse(TfliteTensorsInfo *tensors_info, const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                 const std::unique_ptr<tflite::ModelT> &tflite_model,
                                 const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph, schema::CNodeT *op) {
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  std::unique_ptr<schema::ReduceT> attr = std::make_unique<schema::ReduceT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }

  const auto &tflite_attr = tflite_op->builtin_options.AsReducerOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op: " << op->name << " attr failed";
    return RET_NULL_PTR;
  }
  attr->keepDims = tflite_attr->keep_dims;

  std::vector<std::string> node_name_str;
  Split(op->name, &node_name_str, "-");
  const char *node_name = node_name_str.data()->c_str();

  if (std::strcmp(node_name, "ReduceMax") == 0) {
    MS_LOG(DEBUG) << "parse TfliteReduceMaxParser";
    attr->mode = schema::ReduceMode_ReduceMax;
  } else if (std::strcmp(node_name, "ReduceMin") == 0) {
    MS_LOG(DEBUG) << "parse TfliteReduceMinParser";
    attr->mode = schema::ReduceMode_ReduceMin;
  } else if (std::strcmp(node_name, "ReduceProd") == 0) {
    MS_LOG(DEBUG) << "parse TfliteReduceProdParser";
    attr->mode = schema::ReduceMode_ReduceProd;
  } else if (std::strcmp(node_name, "Sum") == 0) {
    MS_LOG(DEBUG) << "parse TfliteSumParser";
    attr->mode = schema::ReduceMode_ReduceSum;
  } else if (std::strcmp(node_name, "Mean") == 0) {
    MS_LOG(DEBUG) << "parse TfliteMeanParser";
    attr->mode = schema::ReduceMode_ReduceMean;
  } else if (std::strcmp(node_name, "ReduceAny") == 0) {
    // attr->mode;
    MS_LOG(ERROR) << "ms-lite haven't supported REDUCE_ANY now";
    return RET_NOT_SUPPORT;
  }

  if (GetTfliteData(tflite_op->inputs[1], tflite_subgraph->tensors, tflite_model->buffers, attr->axes)) {
    MS_LOG(ERROR) << "get reduce -> axes failed";
    return RET_ERROR;
  }

  op->primitive->value.type = schema::PrimitiveType_Reduce;
  op->primitive->value.value = attr.release();

  AddOpInput(op, tensors_info, tflite_op->inputs[0], tflite_subgraph->tensors.size(), schema::Format::Format_NHWC);
  AddOpOutput(op, tensors_info, tflite_op->outputs[0], tflite_subgraph->tensors.size(), schema::Format::Format_NHWC);
  return RET_OK;
}

TfliteNodeRegister g_TfliteSumParser("Sum", new TfliteSumParser());
TfliteNodeRegister g_TfliteMeanParser("Mean", new TfliteMeanParser());
TfliteNodeRegister g_TfliteReduceMaxParser("ReduceMax", new TfliteReduceMaxParser());
TfliteNodeRegister g_TfliteReduceMinParser("ReduceMin", new TfliteReduceMinParser());
TfliteNodeRegister g_TfliteReduceProdParser("ReduceProd", new TfliteReduceProdParser());
TfliteNodeRegister g_TfliteReduceAnyParser("ReduceAny", new TfliteReduceAnyParser());
}  // namespace lite
}  // namespace mindspore

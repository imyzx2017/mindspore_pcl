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

#include "tools/converter/parser/tflite/tflite_range_parser.h"
#include <vector>
#include <memory>
#include <map>

namespace mindspore {
namespace lite {
STATUS TfliteRangeParser::Parse(TfliteTensorsInfo *tensors_info, const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                const std::unique_ptr<tflite::ModelT> &tflite_model,
                                const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph, schema::CNodeT *op) {
  MS_LOG(DEBUG) << "parse TfliteRangeParser";
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  std::unique_ptr<schema::RangeT> attr = std::make_unique<schema::RangeT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }

  attr->dType = 0;
  std::vector<int> limit;
  std::vector<int> delta;
  int status = GetTfliteData(tflite_op->inputs[1], tflite_subgraph->tensors, tflite_model->buffers, limit);
  if (status != RET_OK && status != RET_NO_CHANGE) {
    MS_LOG(ERROR) << "range -> limit get failed";
    return RET_ERROR;
  } else if (status == RET_OK) {
    status = GetTfliteData(tflite_op->inputs[2], tflite_subgraph->tensors, tflite_model->buffers, delta);
    if (status != RET_OK && status != RET_NO_CHANGE) {
      MS_LOG(ERROR) << "stridedSlice -> end get failed";
      return RET_ERROR;
    }
  }
  if (status == RET_OK) {
    attr->limit = limit.front();
    attr->delta = delta.front();
  }
  op->primitive->value.type = schema::PrimitiveType_Range;
  op->primitive->value.value = attr.release();

  int input_num = status == RET_OK ? 1 : 3;
  for (int i = 0; i < input_num; ++i) {
    AddOpInput(op, tensors_info, tflite_op->inputs[i], tflite_subgraph->tensors.size(), schema::Format::Format_NHWC);
  }
  AddOpOutput(op, tensors_info, tflite_op->outputs[0], tflite_subgraph->tensors.size(), schema::Format::Format_NHWC);
  return RET_OK;
}

TfliteNodeRegister g_tfliteRangeParser("Range", new TfliteRangeParser());
}  // namespace lite
}  // namespace mindspore

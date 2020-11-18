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

#include "tools/converter/parser/onnx/onnx_reduce_parser.h"
#include <memory>

namespace mindspore {
namespace lite {
STATUS OnnxReduceParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node,
                               schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx ReduceParser";
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

  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "axes") {
      const int &size = onnx_node_attr.ints_size();
      for (int i = 0; i < size; ++i) {
        attr->axes.push_back(onnx_node_attr.ints(i));
      }
    } else if (attribute_name == "keepdims") {
      attr->keepDims = static_cast<bool>(onnx_node_attr.i());
    }
  }
  const auto &type = onnx_node.op_type();
  if (type == "ReduceMean") {
    attr->mode = schema::ReduceMode_ReduceMean;
  } else if (type == "ReduceMax") {
    attr->mode = schema::ReduceMode_ReduceMax;
  } else if (type == "ReduceMin") {
    attr->mode = schema::ReduceMode_ReduceMin;
  } else if (type == "ReduceSum") {
    attr->mode = schema::ReduceMode_ReduceSum;
  } else {
    MS_LOG(ERROR) << "unsupported type";
    return RET_ERROR;
  }

  op->primitive->value.type = schema::PrimitiveType_Reduce;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

OnnxNodeRegistrar g_onnxReduceMeanParser("ReduceMean", new OnnxReduceParser());
OnnxNodeRegistrar g_onnxReduceMaxParser("ReduceMax", new OnnxReduceParser());
OnnxNodeRegistrar g_onnxReduceMinParser("ReduceMin", new OnnxReduceParser());
OnnxNodeRegistrar g_onnxReduceProdParser("ReduceProd", new OnnxReduceParser());
OnnxNodeRegistrar g_onnxReduceSumParser("ReduceSum", new OnnxReduceParser());
OnnxNodeRegistrar g_onnxReduceSumSquareParser("ReduceSumSquare", new OnnxReduceParser());
}  // namespace lite
}  // namespace mindspore

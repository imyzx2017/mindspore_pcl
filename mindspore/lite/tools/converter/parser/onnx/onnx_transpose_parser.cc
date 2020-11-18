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

#include "tools/converter/parser/onnx/onnx_transpose_parser.h"
#include <memory>

namespace mindspore {
namespace lite {
STATUS OnnxTransposeParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node,
                                  schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx TransposeParser";
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  std::unique_ptr<schema::TransposeT> attr = std::make_unique<schema::TransposeT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }

  attr->conjugate = false;
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "axes") {
      attr->perm.resize(onnx_node_attr.ints_size());
      for (int i = 0; i < onnx_node_attr.ints_size(); ++i) {
        attr->perm[i] = onnx_node_attr.ints(i);
      }
    }
    if (attribute_name == "perm") {
      attr->perm.resize(onnx_node_attr.ints_size());
      for (int i = 0; i < onnx_node_attr.ints_size(); ++i) {
        attr->perm[i] = onnx_node_attr.ints(i);
      }
    }
  }

  op->primitive->value.type = schema::PrimitiveType_Transpose;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

OnnxNodeRegistrar g_onnxTransposeParser("Transpose", new OnnxTransposeParser());
OnnxNodeRegistrar g_onnxInt8TransposeParser("Int8Transpose", new OnnxTransposeParser());
}  // namespace lite
}  // namespace mindspore

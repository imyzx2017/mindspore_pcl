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

#include "tools/converter/parser/onnx/onnx_instance_norm_parser.h"
#include <memory>

namespace mindspore {
namespace lite {
STATUS OnnxInstanceNormParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node,
                                     schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx InstanceNormParser";
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  std::unique_ptr<schema::InstanceNormT> attr = std::make_unique<schema::InstanceNormT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }

  if (!onnx_node.attribute().empty()) {
    auto onnx_node_attr = onnx_node.attribute().at(0);
    if (onnx_node_attr.name() == "epsilon") {
      attr->epsilon = onnx_node_attr.f();
    }
  }

  op->primitive->value.type = schema::PrimitiveType_InstanceNorm;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

OnnxNodeRegistrar g_onnxInstanceNormParser("InstanceNormalization", new OnnxInstanceNormParser());
}  // namespace lite
}  // namespace mindspore

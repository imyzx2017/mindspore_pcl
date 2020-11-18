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

#include "tools/converter/parser/onnx/onnx_conv_parser.h"
#include <algorithm>
#include <memory>
#include <vector>

namespace mindspore {
namespace lite {
constexpr int32_t kSingleGroup = 1;
bool OnnxConvParser::ParseGroupConvolution(const std::unique_ptr<schema::Conv2DT> &attr, schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx DepthwiseConvParser";
  std::unique_ptr<schema::DepthwiseConv2DT> depthwiseConv2DParam = std::make_unique<schema::DepthwiseConv2DT>();
  if (depthwiseConv2DParam == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return false;
  }
  depthwiseConv2DParam->format = attr->format;
  depthwiseConv2DParam->channelIn = attr->channelIn;
  depthwiseConv2DParam->channelMultiplier = attr->channelOut / attr->channelIn;
  depthwiseConv2DParam->kernelW = attr->kernelW;
  depthwiseConv2DParam->kernelH = attr->kernelH;
  depthwiseConv2DParam->strideW = attr->strideW;
  depthwiseConv2DParam->strideH = attr->strideH;
  depthwiseConv2DParam->padMode = attr->padMode;
  depthwiseConv2DParam->padUp = attr->padUp;
  depthwiseConv2DParam->padDown = attr->padDown;
  depthwiseConv2DParam->padLeft = attr->padLeft;
  depthwiseConv2DParam->padRight = attr->padRight;
  depthwiseConv2DParam->dilateW = attr->dilateW;
  depthwiseConv2DParam->dilateH = attr->dilateH;
  depthwiseConv2DParam->hasBias = attr->hasBias;
  depthwiseConv2DParam->activationType = attr->activationType;

  op->primitive->value.type = schema::PrimitiveType_DepthwiseConv2D;
  op->primitive->value.value = depthwiseConv2DParam.release();
  return true;
}

STATUS OnnxConvParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node, schema::CNodeT *op) {
  MS_LOG(DEBUG) << "onnx ConvParser";
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  std::unique_ptr<schema::Conv2DT> attr = std::make_unique<schema::Conv2DT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }
  // set default params
  attr->strideH = 1;
  attr->strideW = 1;
  attr->dilateH = 1;
  attr->dilateW = 1;
  attr->group = 1;
  attr->padMode = schema::PadMode_NOTSET;
  attr->format = schema::Format::Format_NCHW;
  // set opdef each attr params
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    if (onnx_node_attr.name() == "group") {
      attr->group = static_cast<int32_t>(onnx_node_attr.i());
    } else if (onnx_node_attr.name() == "dilations") {
      if (onnx_node_attr.ints().size() != 2) {
        MS_LOG(ERROR) << "dilations size " << onnx_node_attr.ints().size() << " is not 2";
        return RET_ERROR;
      }
      attr->dilateH = static_cast<int32_t>(onnx_node_attr.ints(0));
      attr->dilateW = static_cast<int32_t>(onnx_node_attr.ints(1));
    } else if (onnx_node_attr.name() == "kernels") {
      if (onnx_node_attr.ints().size() != 2) {
        MS_LOG(ERROR) << "kernel_shape size " << onnx_node_attr.ints().size() << " is not 2";
        return RET_ERROR;
      }
      attr->kernelH = static_cast<int32_t>(onnx_node_attr.ints(0));
      attr->kernelW = static_cast<int32_t>(onnx_node_attr.ints(1));
    } else if (onnx_node_attr.name() == "kernel_shape") {
      if (onnx_node_attr.ints().size() != 2) {
        MS_LOG(ERROR) << "kernel_shape size " << onnx_node_attr.ints().size() << " is not 2";
        return RET_ERROR;
      }
      attr->kernelH = static_cast<int32_t>(onnx_node_attr.ints(0));
      attr->kernelW = static_cast<int32_t>(onnx_node_attr.ints(1));
    } else if (onnx_node_attr.name() == "auto_pad") {
      attr->padMode = GetOnnxPadMode(onnx_node_attr);
    } else if (onnx_node_attr.name() == "pads") {
      if (onnx_node_attr.ints().size() != 4) {
        MS_LOG(ERROR) << "pads size " << onnx_node_attr.ints().size() << " is not 4";
        return RET_ERROR;
      }
      attr->padUp = static_cast<int32_t>(onnx_node_attr.ints(0));
      attr->padLeft = static_cast<int32_t>(onnx_node_attr.ints(1));
      attr->padDown = static_cast<int32_t>(onnx_node_attr.ints(2));
      attr->padRight = static_cast<int32_t>(onnx_node_attr.ints(3));
    } else if (onnx_node_attr.name() == "strides") {
      if (onnx_node_attr.ints().size() != 2) {
        MS_LOG(ERROR) << "strides size " << onnx_node_attr.ints().size() << " is not 2";
        return RET_ERROR;
      }
      attr->strideH = static_cast<int32_t>(onnx_node_attr.ints(0));
      attr->strideW = static_cast<int32_t>(onnx_node_attr.ints(1));
    } else if (onnx_node_attr.name() == "order") {
      if (onnx_node_attr.s() == "NHWC") {
        attr->format = schema::Format::Format_NHWC;
      } else {
        MS_LOG(ERROR) << "Unsupported format: " << onnx_node_attr.s();
        return RET_ERROR;
      }
    }
  }

  const auto &onnx_conv_weight = onnx_node.input(1);
  if (onnx_node.op_type() == "Conv") {
    auto nodeIter =
      std::find_if(onnx_graph.initializer().begin(), onnx_graph.initializer().end(),
                   [onnx_conv_weight](const onnx::TensorProto &proto) { return proto.name() == onnx_conv_weight; });
    if (nodeIter == onnx_graph.initializer().end()) {
      MS_LOG(ERROR) << "not find node: " << onnx_conv_weight;
      return RET_ERROR;
    }
    std::vector<int> weight_shape;
    auto size = (*nodeIter).dims_size();
    for (int i = 0; i < size; ++i) {
      weight_shape.emplace_back((*nodeIter).dims(i));
    }
    attr->channelOut = weight_shape[0];
    attr->channelIn = weight_shape[1] * attr->group;
  } else {
    auto nodeIter =
      std::find_if(onnx_graph.node().begin(), onnx_graph.node().end(),
                   [onnx_conv_weight](const onnx::NodeProto &proto) { return proto.output(0) == onnx_conv_weight; });
    if (nodeIter == onnx_graph.node().end()) {
      MS_LOG(ERROR) << "can not find node: " << onnx_conv_weight;
      return RET_ERROR;
    }
    std::vector<int> dims;
    auto iter = std::find_if((*nodeIter).attribute().begin(), (*nodeIter).attribute().end(),
                             [](const onnx::AttributeProto &attr) { return attr.name() == "shape"; });
    if (iter != (*nodeIter).attribute().end()) {
      MS_ASSERT(iter->ints() != nullptr);
      MS_ASSERT(iter->ints().begin() != nullptr);
      MS_ASSERT(iter->ints().end() != nullptr);
      dims.insert(dims.begin(), iter->ints().begin(), iter->ints().end());
    }
    attr->channelOut = dims[0];
    attr->channelIn = dims[3] * attr->group;
  }
  attr->hasBias = onnx_node.input().size() == 3;
  if (onnx_node.op_type() == "ConvRelu" || onnx_node.op_type() == "Int8ConvRelu") {
    attr->activationType = schema::ActivationType_RELU;
  } else {
    attr->activationType = schema::ActivationType_NO_ACTIVATION;
  }

  if (attr != nullptr && attr->group > kSingleGroup && attr->group == attr->channelIn) {
    if (!ParseGroupConvolution(attr, op)) {
      MS_LOG(ERROR) << "Convert Convolution to Depthwise failed";
      return RET_ERROR;
    }
  } else {
    op->primitive->value.type = schema::PrimitiveType_Conv2D;
    op->primitive->value.value = attr.release();
  }
  return RET_OK;
}

OnnxNodeRegistrar g_onnxConvParser("Conv", new OnnxConvParser());
OnnxNodeRegistrar g_onnxInt8ConvParser("Int8Conv", new OnnxConvParser());
OnnxNodeRegistrar g_onnxConvReluParser("ConvRelu", new OnnxConvParser());
OnnxNodeRegistrar g_onnxInt8ConvReluParser("Int8ConvRelu", new OnnxConvParser());
}  // namespace lite
}  // namespace mindspore

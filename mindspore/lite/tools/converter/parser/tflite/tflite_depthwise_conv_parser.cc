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

#include "tools/converter/parser/tflite/tflite_depthwise_conv_parser.h"
#include <vector>
#include <memory>
#include <map>

namespace mindspore {
namespace lite {
STATUS TfliteDepthwiseConv2DParser::Parse(TfliteTensorsInfo *tensors_info,
                                          const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                          const std::unique_ptr<tflite::ModelT> &tflite_model,
                                          const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                          schema::CNodeT *op) {
  MS_LOG(DEBUG) << "parse TfliteDepthwiseConv2DParser";
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  std::unique_ptr<schema::DepthwiseConv2DT> attr = std::make_unique<schema::DepthwiseConv2DT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }

  const auto &tflite_attr = tflite_op->builtin_options.AsDepthwiseConv2DOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op: " << op->name.c_str() << " attr failed";
    return RET_NULL_PTR;
  }
  attr->strideW = tflite_attr->stride_w;
  attr->strideH = tflite_attr->stride_h;
  attr->dilateH = tflite_attr->dilation_h_factor;
  attr->dilateW = tflite_attr->dilation_w_factor;
  attr->padMode = GetPadMode(tflite_attr->padding);
  attr->format = schema::Format::Format_NHWC;
  attr->activationType = GetActivationFunctionType(tflite_attr->fused_activation_function);
  attr->hasBias = true;
  attr->channelMultiplier = tflite_attr->depth_multiplier;

  // get the data tensor
  auto data_index = tflite_op->inputs[1];
  const auto &data_tensor = tflite_subgraph->tensors[data_index];
  if (data_tensor == nullptr) {
    MS_LOG(ERROR) << "the data tensor is null";
    return RET_NULL_PTR;
  }
  auto data_shape = data_tensor->shape;
  attr->channelIn = data_shape[3];

  // get the weight tensor
  auto weight_index = tflite_op->inputs[1];
  const auto &weight_tensor = tflite_subgraph->tensors[weight_index];
  if (weight_tensor == nullptr) {
    MS_LOG(ERROR) << "the weight tensor is null";
    return RET_NULL_PTR;
  }
  auto weight_shape = weight_tensor->shape;
  attr->kernelH = weight_shape[1];
  attr->kernelW = weight_shape[2];

  // calculate pad params
  std::vector<int> params;
  int status =
    getPaddingParam(data_tensor, attr->padMode, attr->strideH, attr->strideW, attr->kernelH, attr->kernelW, &params);
  if (status != RET_OK && status != RET_NO_CHANGE) {
    MS_LOG(ERROR) << "get padding params failed";
    return RET_ERROR;
  } else if (status == RET_OK) {
    attr->padUp = params.at(0);
    attr->padDown = params.at(1);
    attr->padLeft = params.at(2);
    attr->padRight = params.at(3);
  }

  op->primitive->value.type = schema::PrimitiveType_DepthwiseConv2D;
  op->primitive->value.value = attr.release();

  AddOpInput(op, tensors_info, tflite_op->inputs[0], tflite_subgraph->tensors.size(), schema::Format::Format_NHWC);
  AddOpInput(op, tensors_info, tflite_op->inputs[1], tflite_subgraph->tensors.size(), schema::Format::Format_KHWC);
  AddOpInput(op, tensors_info, tflite_op->inputs[2], tflite_subgraph->tensors.size(), schema::Format::Format_NHWC);
  AddOpOutput(op, tensors_info, tflite_op->outputs[0], tflite_subgraph->tensors.size(), schema::Format::Format_NHWC);
  return RET_OK;
}

TfliteNodeRegister g_tfliteDepthwiseConv2DParser("DepthwiseConv2D", new TfliteDepthwiseConv2DParser());
}  // namespace lite
}  // namespace mindspore

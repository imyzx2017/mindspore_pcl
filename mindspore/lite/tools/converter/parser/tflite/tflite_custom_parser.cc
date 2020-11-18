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

#include "tools/converter/parser/tflite/tflite_custom_parser.h"
#include <map>
#include <memory>
#include <vector>
#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/flexbuffers.h"

namespace mindspore {
namespace lite {
STATUS TfliteCustomParser::DetectPostProcess(const std::vector<uint8_t> &custom_attr, schema::CNodeT *op,
                                             const std::unique_ptr<tflite::OperatorT> &tflite_op) {
  std::unique_ptr<schema::DetectionPostProcessT> attr = std::make_unique<schema::DetectionPostProcessT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }

  auto attr_map = flexbuffers::GetRoot(custom_attr).AsMap();
  attr->format = schema::Format::Format_NHWC;
  attr->inputSize = tflite_op->inputs.size();
  attr->hScale = attr_map["h_scale"].AsFloat();
  attr->wScale = attr_map["w_scale"].AsFloat();
  attr->xScale = attr_map["x_scale"].AsFloat();
  attr->yScale = attr_map["y_scale"].AsFloat();
  attr->NmsIouThreshold = attr_map["nms_iou_threshold"].AsFloat();
  attr->NmsScoreThreshold = attr_map["nms_score_threshold"].AsFloat();
  attr->MaxDetections = attr_map["max_detections"].AsInt32();
  if (attr_map["detections_per_class"].IsNull()) {
    attr->DetectionsPerClass = 100;
  } else {
    attr->DetectionsPerClass = attr_map["detections_per_class"].AsInt32();
  }
  attr->MaxClassesPerDetection = attr_map["max_classes_per_detection"].AsInt32();
  attr->NumClasses = attr_map["num_classes"].AsInt32();
  if (attr_map["use_regular_nms"].IsNull()) {
    attr->UseRegularNms = false;
  } else {
    attr->UseRegularNms = attr_map["use_regular_nms"].AsBool();
  }
  if (attr_map["_output_quantized"].IsNull()) {
    attr->OutQuantized = false;
  } else {
    attr->OutQuantized = attr_map["_output_quantized"].AsBool();
  }

  op->primitive->value.type = schema::PrimitiveType_DetectionPostProcess;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

STATUS TfliteCustomParser::AudioSpectrogram(const std::vector<uint8_t> &custom_attr, schema::CNodeT *op,
                                            const std::unique_ptr<tflite::OperatorT> &tflite_op) {
  std::unique_ptr<schema::AudioSpectrogramT> attr = std::make_unique<schema::AudioSpectrogramT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }
  auto attr_map = flexbuffers::GetRoot(custom_attr).AsMap();
  attr->windowSize = attr_map["window_size"].AsInt64();
  attr->stride = attr_map["stride"].AsInt64();
  attr->magSquare = attr_map["magnitude_squared"].AsBool();

  op->primitive->value.type = schema::PrimitiveType_AudioSpectrogram;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

STATUS TfliteCustomParser::Mfcc(const std::vector<uint8_t> &custom_attr, schema::CNodeT *op,
                                const std::unique_ptr<tflite::OperatorT> &tflite_op) {
  std::unique_ptr<schema::MfccT> attr = std::make_unique<schema::MfccT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }
  auto attr_map = flexbuffers::GetRoot(custom_attr).AsMap();
  attr->freqUpperLimit = attr_map["upper_frequency_limit"].AsInt64();
  attr->freqLowerLimit = attr_map["lower_frequency_limit"].AsInt64();
  attr->filterBankChannelNum = attr_map["filterbank_channel_count"].AsInt64();
  attr->dctCoeffNum = attr_map["dct_coefficient_count"].AsInt64();

  op->primitive->value.type = schema::PrimitiveType_Mfcc;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

STATUS TfliteCustomParser::Predict(const std::vector<uint8_t> &custom_attr, schema::CNodeT *op,
                                   const std::unique_ptr<tflite::OperatorT> &tflite_op) {
  std::unique_ptr<schema::CustomPredictT> attr = std::make_unique<schema::CustomPredictT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }
  attr->outputNum = reinterpret_cast<const int *>(custom_attr.data())[0];
  attr->weightThreshold = reinterpret_cast<const float *>(custom_attr.data())[1];
  op->primitive->value.type = schema::PrimitiveType_CustomPredict;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

STATUS TfliteCustomParser::Normalize(const std::vector<uint8_t> &custom_attr, schema::CNodeT *op,
                                     const std::unique_ptr<tflite::OperatorT> &tflite_op) {
  std::unique_ptr<schema::CustomNormalizeT> attr = std::make_unique<schema::CustomNormalizeT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }
  op->primitive->value.type = schema::PrimitiveType_CustomNormalize;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

STATUS TfliteCustomParser::ExtractFeatures(const std::vector<uint8_t> &custom_attr, schema::CNodeT *op,
                                           const std::unique_ptr<tflite::OperatorT> &tflite_op) {
  std::unique_ptr<schema::CustomExtractFeaturesT> attr = std::make_unique<schema::CustomExtractFeaturesT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }
  op->primitive->value.type = schema::PrimitiveType_CustomExtractFeatures;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

STATUS TfliteCustomParser::Rfft(const std::vector<uint8_t> &custom_attr, schema::CNodeT *op,
                                const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                const std::unique_ptr<tflite::ModelT> &tflite_model,
                                const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph) {
  std::unique_ptr<schema::RfftT> attr = std::make_unique<schema::RfftT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }
  std::vector<int> fft_length;
  if (GetTfliteData(tflite_op->inputs[1], tflite_subgraph->tensors, tflite_model->buffers, fft_length)) {
    MS_LOG(ERROR) << "rfft -> fftLength get failed";
    return RET_ERROR;
  }
  attr->fftLength = fft_length[0];
  op->primitive->value.type = schema::PrimitiveType_Rfft;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

STATUS TfliteCustomParser::FftReal(const std::vector<uint8_t> &custom_attr, schema::CNodeT *op,
                                   const std::unique_ptr<tflite::OperatorT> &tflite_op) {
  std::unique_ptr<schema::FftRealT> attr = std::make_unique<schema::FftRealT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }
  op->primitive->value.type = schema::PrimitiveType_FftReal;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

STATUS TfliteCustomParser::FftImag(const std::vector<uint8_t> &custom_attr, schema::CNodeT *op,
                                   const std::unique_ptr<tflite::OperatorT> &tflite_op) {
  std::unique_ptr<schema::FftImagT> attr = std::make_unique<schema::FftImagT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }
  op->primitive->value.type = schema::PrimitiveType_FftImag;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

STATUS TfliteCustomParser::Parse(TfliteTensorsInfo *tensors_info, const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                 const std::unique_ptr<tflite::ModelT> &tflite_model,
                                 const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph, schema::CNodeT *op) {
  MS_LOG(DEBUG) << "parse TfliteCustomParser";
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }
  const auto &custom_attr = tflite_op->custom_options;
  const auto &opcode_index = tflite_op->opcode_index;
  const auto &custom_type = tflite_model->operator_codes[opcode_index]->custom_code;
  int status = RET_OK;
  if (custom_type == "TFLite_Detection_PostProcess") {
    status = DetectPostProcess(custom_attr, op, tflite_op);
  } else if (custom_type == "Predict") {
    status = Predict(custom_attr, op, tflite_op);
  } else if (custom_type == "Normalize") {
    status = Normalize(custom_attr, op, tflite_op);
  } else if (custom_type == "ExtractFeatures") {
    status = ExtractFeatures(custom_attr, op, tflite_op);
  } else if (custom_type == "AudioSpectrogram") {
    status = AudioSpectrogram(custom_attr, op, tflite_op);
  } else if (custom_type == "Mfcc") {
    status = Mfcc(custom_attr, op, tflite_op);
  } else if (custom_type == "FlexRFFT") {
    status = Rfft(custom_attr, op, tflite_op, tflite_model, tflite_subgraph);
  } else if (custom_type == "FlexReal") {
    status = FftReal(custom_attr, op, tflite_op);
  } else if (custom_type == "FlexImag") {
    status = FftImag(custom_attr, op, tflite_op);
  } else {
    MS_LOG(ERROR) << "the custom op hasn't been supported now";
    status = RET_NOT_FIND_OP;
  }
  if (status != RET_OK) {
    return status;
  }
  for (size_t i = 0; i < tflite_op->inputs.size(); ++i) {
    AddOpInput(op, tensors_info, tflite_op->inputs[i], tflite_subgraph->tensors.size(), schema::Format::Format_NHWC);
  }
  for (size_t i = 0; i < tflite_op->outputs.size(); ++i) {
    AddOpOutput(op, tensors_info, tflite_op->outputs[i], tflite_subgraph->tensors.size(), schema::Format::Format_NHWC);
  }
  return status;
}

TfliteNodeRegister g_tfliteCustomParser("Custom", new TfliteCustomParser());
}  // namespace lite
}  // namespace mindspore

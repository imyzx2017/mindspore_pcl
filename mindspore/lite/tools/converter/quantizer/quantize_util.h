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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANTIZER_UTIL_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANTIZER_UTIL_H

#include <memory>
#include <string>
#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <limits>
#include <utility>
#include "tools/converter/quantizer/quantizer.h"
#include "src/ops/primitive_c.h"
#include "include/errorcode.h"
#include "ir/func_graph.h"
#include "ir/anf.h"
#include "include/model.h"
#include "base/base.h"
#include "ir/primitive.h"
#include "abstract/dshape.h"
#include "tools/converter/quantizer/bitpacking.h"

namespace mindspore {
namespace lite {
namespace quant {
static constexpr size_t UINT8_QUANTIZATION = 8;
static constexpr size_t WEIGHT_INDEX = 1;

/**
 * 1. when op's weight size > mWeightSize just skip
 * 2. only do conv/deconv/convdepthwise/deconvdepthwise/mul/matmul/batchmatmul quantization
 * 3. when conv/deconv/convdepthwise/deconvdepthwise ops' weight channel size > covWeightQuantChannelThreshold just skip
 * */
class QuantStrategy {
 public:
  explicit QuantStrategy(size_t weightSize, size_t covWeightQuantChannelThreshold = 16);

  ~QuantStrategy() = default;

  bool CanConvOpQuantized(const CNodePtr &node) const;
  bool CanMulOpQuantized(const CNodePtr &node) const;
  bool CanOpPostQuantized(AnfNodePtr &node) const;

 private:
  size_t mWeightSize;
  size_t mConvWeightQuantChannelThreshold;
  static const std::vector<schema::PrimitiveType> conv_types;
  static const std::vector<schema::PrimitiveType> mul_types;
};

constexpr float delta = 0.1;
constexpr float ratio = 10.0;
constexpr int percent = 10;

STATUS CalQuantizationParams(schema::QuantParamT *quantParam, double mMin, double mMax, bool narrowRange, int quant_max,
                             int quant_min, int num_bits);

STATUS CalQuantizationParams(schema::QuantParamT *quantParam, double mMin, double mMax, bool narrowRange = false,
                             int numBits = UINT8_QUANTIZATION);

std::pair<float, float> OutlierMethod(std::vector<float> min_datas, std::vector<float> max_datas);

std::vector<int8_t> KMeans(float *data, size_t elem_count, size_t k, size_t epochs, schema::QuantParamT *quantParam);

template <typename T>
T QuantizeData(const float originData, const schema::QuantParamT *quantParam) {
  MS_ASSERT(quantParam != nullptr);
  MS_ASSERT(quantParam->inited);
  const auto scale = quantParam->scale;
  const auto zeroPoint = quantParam->zeroPoint;
  const auto numBit = quantParam->numBits;
  const auto narrowRange = quantParam->narrowRange;
  double maxLimitTemp = static_cast<float>((1 << (unsigned int)numBit) - 1);
  const double maxLimit = static_cast<float>(maxLimitTemp - zeroPoint + std::numeric_limits<T>::min()) * scale;
  double minLimit;
  if (narrowRange) {
    minLimit = static_cast<float>(std::numeric_limits<T>::min() + 1 - zeroPoint) * scale;
  } else {
    minLimit = static_cast<float>(std::numeric_limits<T>::min() - zeroPoint) * scale;
  }

  return [maxLimit, minLimit, zeroPoint, scale, narrowRange, originData] {
    double tmp = 0.0f;
    if (originData > maxLimit) {
      tmp = maxLimit;
    } else if (originData < minLimit) {
      tmp = minLimit;
    } else {
      tmp = originData;
    }
    auto quantData = static_cast<T>(std::round(zeroPoint + tmp / scale));
    return quantData;
  }();
}

template <typename T>
T QuantizeData(float originData, const schema::QuantParamT &quantParam, int quant_max, int quant_min) {
  MS_ASSERT(quantParam != nullptr);
  MS_ASSERT(quantParam->inited);
  const auto scale = quantParam.scale;
  const int zeroPoint = quantParam.zeroPoint;
  const auto narrowRange = quantParam.narrowRange;
  const int maxLimit = quant_max;
  const int minLimit = quant_min;

  return [maxLimit, minLimit, zeroPoint, scale, narrowRange, originData] {
    auto quant_data = std::round(originData / scale + zeroPoint);
    if (quant_data > maxLimit) {
      quant_data = maxLimit;
    } else if (quant_data < minLimit) {
      quant_data = minLimit;
    }
    return static_cast<T>(quant_data);
  }();
}
template <typename T>
STATUS QuantFilter(ParamValueLitePtr weight, std::shared_ptr<PrimitiveC> primitive_c, QuantType quantType,
                   int quant_max, int quant_min, size_t bitNum, bool per_channel, bool k_means = false) {
  auto dims = weight->tensor_shape();
  auto op_type = (schema::PrimitiveType)primitive_c->Type();
  if (per_channel) {
    if (dims.size() != 4 && dims.size() != 2 && op_type != schema::PrimitiveType_MatMul) {
      MS_LOG(INFO) << "weight dims size: " << dims.size() << " switch to per-layer quant mode.";
      per_channel = false;
    } else {
      if (dims.size() == 2 && op_type != schema::PrimitiveType_FullConnection) {
        MS_LOG(INFO) << "weight dims size is 2 but op_type is not FullConnection, switch to per-layer quant mode.";
        per_channel = false;
      }
      uint32_t channels = dims[0];
      if (channels == 0) {
        MS_LOG(ERROR) << "channels is 0";
        return RET_ERROR;
      }
    }
  }

  std::vector<schema::QuantParamT> quant_params;
  size_t elem_count = weight->tensor_shape_size();
  auto *raw_datas = static_cast<float *>(weight->tensor_addr());
  if (raw_datas == nullptr) {
    MS_LOG(ERROR) << "rawDatas is nullptr";
    return RET_ERROR;
  }
  std::vector<T> quant_datas(elem_count);
  std::vector<float> dequant_datas(elem_count);
  if (per_channel) {
    // notice: assume Con2D\DepthwiseConv2D's weight format are same: KHWC
    // channel at first
    auto channels = dims[0];
    if (channels == 0) {
      MS_LOG(ERROR) << "channels is zero";
      return RET_ERROR;
    }
    size_t one_filter_size = elem_count / channels;

    for (int i = 0; i < channels; i++) {
      float min = FLT_MAX;
      float max = -FLT_MAX;
      // find min and max
      for (size_t j = 0; j < one_filter_size; j++) {
        auto index = j + i * one_filter_size;
        if (index >= elem_count) {
          MS_LOG(ERROR) << "over flow!";
          return RET_ERROR;
        }
        min = std::min(min, raw_datas[index]);
        max = std::max(max, raw_datas[index]);
      }
      schema::QuantParamT quant_param;
      STATUS status = CalQuantizationParams(&quant_param, min, max, false, quant_max, quant_min, bitNum);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "CalQuantizationParams failed" << status;
        return status;
      }
      // do quantization
      double average_dequant = 0;
      double average_raw = 0;
      for (uint32_t j = 0; j < one_filter_size; j++) {
        auto index = j + i * one_filter_size;
        if (index >= elem_count) {
          MS_LOG(ERROR) << "over flow!";
          return RET_ERROR;
        }
        float raw_data = raw_datas[index];
        auto quant_data = QuantizeData<T>(raw_data, quant_param, quant_max, quant_min);
        quant_datas[index] = quant_data;

        if (quantType == QuantType_WeightQuant) {
          float dequant_data = quant_param.scale * (quant_data - quant_param.zeroPoint);
          dequant_datas[index] = dequant_data;
          average_dequant += dequant_data;
          average_raw += raw_data;
        }
      }
      if (quantType == QuantType_WeightQuant && !k_means) {
        // mean
        average_dequant = average_dequant / one_filter_size;
        average_raw = average_raw / one_filter_size;
        // std
        double variance_dequant = 0;
        double variance_raw = 0;
        for (uint32_t j = 0; j < one_filter_size; j++) {
          auto index = j + i * one_filter_size;
          if (index >= elem_count) {
            MS_LOG(ERROR) << "over flow!";
            return RET_ERROR;
          }
          variance_dequant += std::pow(dequant_datas[index] - average_dequant, 2);
          variance_raw += std::pow(raw_datas[index] - average_raw, 2);
        }
        variance_dequant = std::sqrt(variance_dequant / one_filter_size);
        variance_raw = std::sqrt(variance_raw / one_filter_size);
        quant_param.varCorr = 1;
        if (variance_raw != 0 && variance_dequant != 0) {
          auto temp_var_corr = variance_raw / variance_dequant;
          if (temp_var_corr > 0 && temp_var_corr < 10) {
            quant_param.varCorr = temp_var_corr;
          } else {
            MS_LOG(WARNING) << "unexpected var_corr: " << temp_var_corr;
          }
        }
        quant_param.meanCorr = average_raw - average_dequant * quant_param.varCorr;
      }
      quant_params.emplace_back(quant_param);
    }
    auto ret = memcpy_s(raw_datas, weight->tensor_size(), quant_datas.data(), elem_count * sizeof(T));
    if (ret != EOK) {
      MS_LOG(ERROR) << "memcpy error: " << ret;
      return RET_ERROR;
    }
    weight->set_tensor_size(elem_count * sizeof(T));
  } else {
    // per layer
    float min = FLT_MAX;
    float max = -FLT_MIN;
    for (uint32_t i = 0; i < elem_count; i++) {
      // find max min
      min = std::min(min, raw_datas[i]);
      max = std::max(max, raw_datas[i]);
    }

    schema::QuantParamT quant_param;
    if (!k_means) {
      STATUS status = CalQuantizationParams(&quant_param, min, max, false, quant_max, quant_min, bitNum);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "CalQuantizationParams failed" << status;
        return status;
      }
    }
    quant_params.emplace_back(quant_param);
    // update data and datatype
    for (uint32_t i = 0; i < elem_count; i++) {
      float raw_data = raw_datas[i];
      if (!k_means) {
        auto quant_data = QuantizeData<T>(raw_data, quant_param, quant_max, quant_min);
        quant_datas[i] = quant_data;
      }
    }
    auto ret = memcpy_s(raw_datas, weight->tensor_size(), quant_datas.data(), elem_count * sizeof(T));
    if (ret != EOK) {
      MS_LOG(ERROR) << "memcpy error: " << ret;
      return RET_ERROR;
    }
    weight->set_tensor_size(elem_count * sizeof(T));
  }

  // do bit pack
  if (bitNum != 8 && bitNum != 16) {
    std::vector<T> data{};
    for (size_t i = 0; i < quant_datas.size(); ++i) {
      data.emplace_back((static_cast<T>(quant_datas[i])));
    }
    if (bitNum > 0 && bitNum < 8) {
      std::vector<uint8_t> pack_data{};
      BitPack::BitPacking<T, uint8_t>(bitNum, data, &pack_data);
      auto ret = memcpy_s(raw_datas, weight->tensor_size(), pack_data.data(), pack_data.size() * sizeof(uint8_t));
      if (ret != EOK) {
        MS_LOG(ERROR) << "PostBitPack memcpy_s qDatas_packed failed";
        return RET_ERROR;
      }
      weight->set_tensor_size(pack_data.size() * sizeof(uint8_t));
    } else if (bitNum > 8 && bitNum < 16) {
      std::vector<uint16_t> pack_data{};
      BitPack::BitPacking<T, uint16_t>(bitNum, data, &pack_data);
      auto ret = memcpy_s(raw_datas, weight->tensor_size(), pack_data.data(), pack_data.size() * sizeof(uint16_t));
      if (ret != EOK) {
        MS_LOG(ERROR) << "PostBitPack memcpy_s qDatas_packed failed";
        return RET_ERROR;
      }
      weight->set_tensor_size(pack_data.size() * sizeof(uint16_t));
    }
  }

  if (quant_params.empty()) {
    MS_LOG(ERROR) << "quant_params empty";
    return RET_ERROR;
  }
  if (quantType == QuantType_PostTraining) {
    primitive_c->AddInputQuantParam(quant_params);
  } else {
    primitive_c->SetInputQuantParam(WEIGHT_INDEX, quant_params);
  }
  return RET_OK;
}

schema::PrimitiveType NodePrimitiveType(CNodePtr cnode);
}  // namespace quant
}  // namespace lite
}  // namespace mindspore
#endif

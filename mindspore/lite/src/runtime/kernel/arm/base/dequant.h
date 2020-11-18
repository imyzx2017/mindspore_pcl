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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_DEQUANT_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_DEQUANT_H_

#include <vector>
#include <queue>
#include <cmath>
#include "src/lite_kernel.h"
#include "src/common/utils.h"
#include "src/tensor.h"

namespace mindspore::kernel {
class DequantUtil {
 public:
  static float *DequantWeight(lite::Tensor *input_tensor);

  static void UnPackToInt(const schema::Tensor *input_tensor, void *weight_unpack_data);

  template <typename T>
  static float *DequantData(lite::Tensor *input_tensor) {
    const auto *quant_datas = static_cast<const T *>(input_tensor->MutableData());
    if (quant_datas == nullptr) {
      MS_LOG(ERROR) << "Get quant tensor failed.";
      return nullptr;
    }
    auto *dequant_datas = static_cast<float *>(malloc(input_tensor->ElementsNum() * sizeof(float)));
    if (dequant_datas == nullptr) {
      MS_LOG(ERROR) << "Malloc failed.";
      return nullptr;
    }
    if (input_tensor->shape().size() == kPerBatch &&
        input_tensor->GetQuantParams().size() == static_cast<size_t>(input_tensor->shape()[0])) {  // per batch matmul
      auto per_batch_size = input_tensor->shape()[0];
      auto quant_param = input_tensor->GetQuantParams();
      for (int i = 0; i < per_batch_size; i++) {
        auto param = quant_param.at(i);
        auto scale = param.scale;
        auto zero_point = param.zeroPoint;
        auto matrix_size = input_tensor->ElementsNum() / per_batch_size;
        for (int64_t j = 0; j < matrix_size; j++) {
          dequant_datas[i * matrix_size + j] =
            static_cast<float>((quant_datas[i * matrix_size + j] - zero_point) * scale);
        }
      }
    } else if (input_tensor->GetQuantParams().size() != kPerTensor) {
      size_t channels = static_cast<size_t>(input_tensor->Batch());
      if (input_tensor->GetQuantParams().size() != channels) {
        MS_LOG(ERROR) << "Quant param not equal channel num " << input_tensor->GetQuantParams().size() << channels;
        free(dequant_datas);
        return nullptr;
      }
      size_t per_channel_size = input_tensor->ElementsNum() / channels;
      auto quant_param = input_tensor->GetQuantParams();
      for (size_t i = 0; i < channels; i++) {
        auto param = quant_param.at(i);
        auto scale = param.scale;
        auto zero_point = param.zeroPoint;
        auto var_corr = param.var_corr;
        auto mean_corr = param.mean_corr;
        if (var_corr < 0 || var_corr > 10) {
          MS_LOG(WARNING) << "unexpeted var_corr: " << var_corr;
          var_corr = 1;
        }
        for (size_t j = 0; j < per_channel_size; j++) {
          auto dequant_data = (quant_datas[per_channel_size * i + j] - zero_point) * scale;
          dequant_datas[per_channel_size * i + j] = static_cast<float>(dequant_data * var_corr + mean_corr);
        }
      }
    } else {
      auto quant_param = input_tensor->GetQuantParams();
      auto quant_clusters = input_tensor->GetQuantClusters();
      auto param = quant_param.front();
      auto scale = param.scale;
      auto zero_point = param.zeroPoint;
      for (int64_t j = 0; j < input_tensor->ElementsNum(); j++) {
        if (!quant_clusters.empty()) {
          int8_t index = quant_datas[j];
          if (index > INT8_MAX || index < INT8_MIN) {
            MS_LOG(ERROR) << "KMeans param quant is error.";
            free(dequant_datas);
            return nullptr;
          }
          dequant_datas[j] = static_cast<float>(param.clusters[index - INT8_MIN]);
        } else {
          dequant_datas[j] = static_cast<float>((quant_datas[j] - zero_point) * scale);
        }
      }
    }
    return dequant_datas;
  }

 private:
  template <typename T1, typename T2>
  static void UnPackData(int origin_bit, const T2 &packed_data, std::queue<bool> *unpack_bit_data, void *unpack_int,
                         size_t *count, bool is_last) {
    T2 uint_result = 0;
    T1 result = 0;
    UnPackFromUintToOrigin<T2>(packed_data, unpack_bit_data);
    while (static_cast<int>(unpack_bit_data->size()) >= origin_bit) {
      for (int k = 0; k < origin_bit; k++) {
        bool bit_tmp = unpack_bit_data->front();
        uint_result = (static_cast<int>(bit_tmp) << k) + uint_result;
        unpack_bit_data->pop();
      }
      result = uint_result - static_cast<T2>(pow(2, origin_bit - 1));
      (static_cast<T1 *>(unpack_int))[*count] = result;
      uint_result = 0;
      (*count)++;
    }
    size_t remainder = unpack_bit_data->size();
    if (is_last && remainder > 0) {
      for (size_t i = 0; i < remainder; i++) {
        bool bit = unpack_bit_data->front();
        uint_result = (static_cast<int>(bit) << i) + uint_result;
        unpack_bit_data->pop();
      }
      result = static_cast<T1>(uint_result - static_cast<T2>(pow(2, origin_bit - 1)));
      (static_cast<T1 *>(unpack_int))[*count] = result;
    }
  }

  template <typename T1, typename T2>
  static void UnPackUtil(const schema::Tensor *input_tensor, int origin_bit, void *unpack_int_data) {
    auto weight_data = input_tensor->data()->data();
    int pack_size =
      input_tensor->dataType() == kNumberTypeInt8 ? input_tensor->data()->size() : input_tensor->data()->size() / 2;
    std::queue<bool> unpack_bit_data;
    size_t count = 0;
    for (int i = 0; i < pack_size; ++i) {
      T2 pack_data = (static_cast<const T2 *>(static_cast<const void *>(weight_data)))[i];
      bool is_last = i == pack_size - 1;
      UnPackData<T1, T2>(origin_bit, pack_data, &unpack_bit_data, unpack_int_data, &count, is_last);
    }
  }

  template <typename T2>
  static void UnPackFromUintToOrigin(const T2 &packed_data, std::queue<bool> *unpack_bit_data) {
    auto n = packed_data;
    size_t bit_count = 0;
    while (bit_count < sizeof(T2) * 8) {
      bool a = n % 2;
      n = n >> 1;
      bit_count++;
      unpack_bit_data->push(a);
    }
  }
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_DEQUANT_H_

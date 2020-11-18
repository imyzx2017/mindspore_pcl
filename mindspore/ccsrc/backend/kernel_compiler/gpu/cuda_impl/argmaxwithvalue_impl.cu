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

#include "argmaxwithvalue_impl.cuh"
#include "runtime/device/gpu/cuda_common.h"
#include "include/cuda_fp16.h"
template <typename T, typename S>
__global__ void ArgmaxWithValue(const T *input, const size_t bound, size_t outerSize,
                                size_t innerSize, S *index, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < outerSize * innerSize;
       pos += gridDim.x * blockDim.x) {
    size_t x = pos / innerSize % outerSize;
    size_t y = pos % innerSize;
    S idx = 0;
    size_t InputOffset = x * bound * innerSize + 0 * innerSize + y;
    T maxData = input[InputOffset];
    for (size_t i = 0; i < bound; i++) {
      InputOffset = x * bound * innerSize + i * innerSize + y;
      auto inputData = input[InputOffset];
      idx = inputData > maxData ? i : idx;
      maxData = inputData > maxData ? inputData : maxData;
    }
    output[pos] = maxData;
    index[pos] = idx;
  }
  return;
}

template <typename T, typename S>
void CalArgmaxWithValue(const T *input, const size_t bound_, const size_t outerSize_, const size_t innerSize_,
                        S *index, T *output, cudaStream_t cuda_stream) {
  ArgmaxWithValue<<<GET_BLOCKS(outerSize_), GET_THREADS, 0, cuda_stream>>>(input, bound_, outerSize_, innerSize_,
                                                                           index, output);
  return;
}

template void CalArgmaxWithValue<float, int>(const float *input, const size_t bound_, const size_t outerSize_,
                                             const size_t innerSize_, int *index, float *output,
                                             cudaStream_t cuda_stream);
template void CalArgmaxWithValue<half, int>(const half *input, const size_t bound_, const size_t outerSize_,
                                            const size_t innerSize_, int *index, half *output,
                                            cudaStream_t cuda_stream);

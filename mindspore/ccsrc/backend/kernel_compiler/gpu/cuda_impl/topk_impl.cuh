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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_TOPK_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_TOPK_H_

#include <cuda_runtime.h>
#include "runtime/device/gpu/cuda_common.h"

template <typename T, typename S>
void TopK(const int &outer, const int &inner, const T *input_addr, const S *k, T *output, S *indices, T *data_buff,
          S *index_buff, cudaStream_t stream);

template <typename T, typename S>
void BitonicSortByKey(const int &outer, const int &inner, T *input, S *indices, T *data_buff, S *index_buff,
                      cudaStream_t stream);
int RoundUpPower2(int v);

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_TOPK_H_

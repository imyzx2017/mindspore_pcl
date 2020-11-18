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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_UNSORT_SEGMENT_SUM_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_UNSORT_SEGMENT_SUM_H_

#include <cuda_runtime.h>
#include "runtime/device/gpu/cuda_common.h"

template<typename T, typename S>
void UnsortedSegmentSum(size_t input_dim0, size_t input_dim1, size_t output_dim0, size_t output_dim1,
                        T* input_addr, S* ids, T* output_addr, cudaStream_t stream);

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_UNSORT_SEGMENT_SUM_H_

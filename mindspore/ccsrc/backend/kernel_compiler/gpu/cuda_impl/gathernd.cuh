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

#ifndef MINDSPORE_GATHERND_GPU_CU_H
#define MINDSPORE_GATHERND_GPU_CU_H

#include "runtime/device/gpu/cuda_common.h"

template <typename T, typename S>
void GatherNd(T *input, S *indices, T *output, const size_t &output_dim0, const size_t &output_dim1,
              const size_t &indices_dim1, S *batch_indices, S *batch_strides, cudaStream_t stream);

#endif  // MINDSPORE_GATHERND_GPU_CU_H

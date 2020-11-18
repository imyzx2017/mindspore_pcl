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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_SMOOTH_L1_LOSS_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_SMOOTH_L1_LOSS_H_
template <typename T>
void SmoothL1Loss(const int &input_size, const float &beta, const T *prediction, const T *target, T *loss,
                  cudaStream_t stream);
template <typename T>
void SmoothL1LossGrad(const int &input_size, const float &beta, const T *prediction, const T *target, const T *dloss,
                      T *dx, cudaStream_t stream);
#endif  // MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_SMOOTH_L1_LOSS_H_

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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_MAXPOOLWITHARGMAX_GRAD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_MAXPOOLWITHARGMAX_GRAD_H_
template <typename T, typename S>
void CalMaxPoolWithArgmaxGrad(const T* x, const T* dy, const S* index, const int n, const int c, const int xHeight,
                              const int xWidth, const int dyHeight, const int dyWidth, const int windowHeight,
                              const int windowWidth, const int strideHeight, const int strideWidth, const int padTop,
                              const int padLeft, T* dx, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_MAXPOOLWITHARGMAX_GRAD_H_

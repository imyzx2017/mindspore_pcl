/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/gpu/nn/flatten_gpu_kernel.h"

namespace mindspore {
namespace kernel {
// float64
MS_REG_GPU_KERNEL_ONE(Flatten, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      FlattenGpuFwdKernel, double)
MS_REG_GPU_KERNEL_ONE(Reshape, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      FlattenGpuFwdKernel, double)
MS_REG_GPU_KERNEL_ONE(ExpandDims, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      FlattenGpuFwdKernel, double)
// float32
MS_REG_GPU_KERNEL_ONE(Flatten, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      FlattenGpuFwdKernel, float)
MS_REG_GPU_KERNEL_ONE(Reshape, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      FlattenGpuFwdKernel, float)
MS_REG_GPU_KERNEL_ONE(ExpandDims, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      FlattenGpuFwdKernel, float)
// float16
MS_REG_GPU_KERNEL_ONE(Flatten, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      FlattenGpuFwdKernel, half)
MS_REG_GPU_KERNEL_ONE(ExpandDims, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      FlattenGpuFwdKernel, half)
MS_REG_GPU_KERNEL_ONE(Reshape, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      FlattenGpuFwdKernel, half)
// int64
MS_REG_GPU_KERNEL_ONE(Flatten, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                      FlattenGpuFwdKernel, int64_t)
MS_REG_GPU_KERNEL_ONE(Reshape, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                      FlattenGpuFwdKernel, int64_t)
MS_REG_GPU_KERNEL_ONE(ExpandDims, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                      FlattenGpuFwdKernel, int64_t)
// int32
MS_REG_GPU_KERNEL_ONE(Flatten, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      FlattenGpuFwdKernel, int)
MS_REG_GPU_KERNEL_ONE(Reshape, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      FlattenGpuFwdKernel, int)
MS_REG_GPU_KERNEL_ONE(ExpandDims, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      FlattenGpuFwdKernel, int)
// int16
MS_REG_GPU_KERNEL_ONE(Flatten, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                      FlattenGpuFwdKernel, int16_t)
MS_REG_GPU_KERNEL_ONE(Reshape, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                      FlattenGpuFwdKernel, int16_t)
MS_REG_GPU_KERNEL_ONE(ExpandDims, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                      FlattenGpuFwdKernel, int16_t)
// uint8
MS_REG_GPU_KERNEL_ONE(Flatten, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                      FlattenGpuFwdKernel, uchar)
MS_REG_GPU_KERNEL_ONE(Reshape, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                      FlattenGpuFwdKernel, uchar)
MS_REG_GPU_KERNEL_ONE(ExpandDims, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                      FlattenGpuFwdKernel, uchar)
// int8
MS_REG_GPU_KERNEL_ONE(Flatten, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                      FlattenGpuFwdKernel, char)
MS_REG_GPU_KERNEL_ONE(Reshape, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                      FlattenGpuFwdKernel, char)
MS_REG_GPU_KERNEL_ONE(ExpandDims, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                      FlattenGpuFwdKernel, char)
// bool
MS_REG_GPU_KERNEL_ONE(Flatten, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                      FlattenGpuFwdKernel, bool)
MS_REG_GPU_KERNEL_ONE(Reshape, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                      FlattenGpuFwdKernel, bool)
MS_REG_GPU_KERNEL_ONE(ExpandDims, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                      FlattenGpuFwdKernel, bool)
}  // namespace kernel
}  // namespace mindspore

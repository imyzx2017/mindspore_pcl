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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MUL_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MUL_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include "backend/kernel_compiler/cpu/mkldnn/mkl_cpu_kernel.h"

namespace mindspore {
namespace kernel {
class MulCPUKernel : public MKLCPUKernel {
 public:
  MulCPUKernel() = default;
  ~MulCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  bool need_swap_{false};
};

MS_REG_CPU_KERNEL(
  Mul, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  MulCPUKernel);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MUL_CPU_KERNEL_H_

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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ARITHMETIC_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ARITHMETIC_CPU_KERNEL_H_
#include <memory>
#include <vector>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class ArithmeticCPUKernel : public CPUKernel {
 public:
  ArithmeticCPUKernel() = default;
  ~ArithmeticCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;
  template <typename T>
  void LaunchLess(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
  template <typename T>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

 private:
  void GenIndex(size_t num, std::vector<size_t> *tmp);
  template <typename T>
  void Sub(const T *input1, const T *input2, T *out, size_t start, size_t end);
  template <typename T>
  void Add(const T *input1, const T *input2, T *out, size_t start, size_t end);
  template <typename T>
  void Mul(const T *input1, const T *input2, T *out, size_t start, size_t end);
  template <typename T>
  void RealDiv(const T *input1, const T *input2, T *out, size_t start, size_t end);
  template <typename T>
  void Pow(const T *input1, const T *input2, T *out, size_t start, size_t end);
  template <typename T>
  void AssignAdd(T *input1, const T *input2, T *out, size_t start, size_t end);
  template <typename T>
  void Less(const T *input1, const T *input2, bool *out, size_t start, size_t end);
  std::vector<size_t> input_shape0_;
  std::vector<size_t> input_shape1_;
  std::vector<size_t> input_element_num0_;
  std::vector<size_t> input_element_num1_;
  std::vector<size_t> output_shape_;
  std::vector<size_t> output_element_num_;
  OperateType operate_type_{ADD};
  TypeId dtype_{kTypeUnknown};
};

MS_REG_CPU_KERNEL(
  Sub, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  ArithmeticCPUKernel);
MS_REG_CPU_KERNEL(
  Sub, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  ArithmeticCPUKernel);
MS_REG_CPU_KERNEL(
  Sub, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  ArithmeticCPUKernel);
MS_REG_CPU_KERNEL(
  Pow, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  ArithmeticCPUKernel);
MS_REG_CPU_KERNEL(
  Pow, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  ArithmeticCPUKernel);
MS_REG_CPU_KERNEL(
  Pow, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  ArithmeticCPUKernel);
MS_REG_CPU_KERNEL(
  RealDiv, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  ArithmeticCPUKernel);
MS_REG_CPU_KERNEL(
  RealDiv,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  ArithmeticCPUKernel);
MS_REG_CPU_KERNEL(
  RealDiv, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  ArithmeticCPUKernel);
MS_REG_CPU_KERNEL(
  Less, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
  ArithmeticCPUKernel);
MS_REG_CPU_KERNEL(
  Less, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
  ArithmeticCPUKernel);
MS_REG_CPU_KERNEL(
  Less, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
  ArithmeticCPUKernel);
MS_REG_CPU_KERNEL(
  AssignAdd, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  ArithmeticCPUKernel);
MS_REG_CPU_KERNEL(
  AssignAdd, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  ArithmeticCPUKernel);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ARITHMETIC_CPU_KERNEL_H_

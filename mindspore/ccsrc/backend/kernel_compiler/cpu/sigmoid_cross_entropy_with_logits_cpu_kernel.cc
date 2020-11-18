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

#include "backend/kernel_compiler/cpu/sigmoid_cross_entropy_with_logits_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
void SigmoidCrossEntropyWithLogitsCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  CheckParam(kernel_node);
  dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0);
  std::vector<uint64_t> x_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  for (const uint64_t &d : x_shape) {
    tensor_size_ *= d;
  }
}

bool SigmoidCrossEntropyWithLogitsCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                                    const std::vector<kernel::AddressPtr> &,
                                                    const std::vector<kernel::AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  }
  return true;
}

template <typename T>
void SigmoidCrossEntropyWithLogitsCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                          const std::vector<AddressPtr> &outputs) {
  auto logits_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto labels_addr = reinterpret_cast<T *>(inputs[1]->addr);
  auto output_addr = reinterpret_cast<T *>(outputs[0]->addr);
  T zero = (T)0.0;
  T one = (T)1.0;
  T two = (T)2.0;
  for (uint64_t i = 0; i < tensor_size_; ++i) {
    if (logits_addr[i] >= zero) {
      output_addr[i] = log1p(exp(logits_addr[i] - two * logits_addr[i])) - logits_addr[i] * (labels_addr[i] - one);
    } else {
      output_addr[i] = log1p(exp(logits_addr[i])) - logits_addr[i] * labels_addr[i];
    }
  }
}

void SigmoidCrossEntropyWithLogitsCPUKernel::CheckParam(const CNodePtr &kernel_node) {
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 2) {
    MS_LOG(EXCEPTION) << "SigmoidCrossEntropyWithLogitsCPUKernel needs 2 inputs, but gets " << input_num;
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "SigmoidCrossEntropyWithLogitsCPUKernel expects 1 output, but gets" << output_num;
  }
}
}  // namespace kernel
}  // namespace mindspore

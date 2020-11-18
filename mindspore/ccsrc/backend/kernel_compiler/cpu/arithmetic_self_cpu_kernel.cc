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
#include <cmath>
#include <string>
#include <thread>
#include "backend/kernel_compiler/cpu/arithmetic_self_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
template <typename T>
void Square(const T *in, T *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    out[i] = in[i] * in[i];
  }
}

template <typename T>
void Neg(const T *in, T *out, size_t start, size_t end) {
  for (size_t i = start; i < end; i++) {
    out[i] = -in[i];
  }
}
}  // namespace

void ArithmeticSelfCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::string kernel_name = AnfAlgo::GetCNodeName(kernel_node);
  if (kernel_name == prim::kPrimSquare->name()) {
    operate_type_ = SQUARE;
  } else if (kernel_name == prim::kPrimNeg->name()) {
    operate_type_ = NEG;
  }
  dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0);
}

bool ArithmeticSelfCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                     const std::vector<kernel::AddressPtr> & /*workspace*/,
                                     const std::vector<kernel::AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeFloat32 || dtype_ == kNumberTypeFloat16 || dtype_ == kNumberTypeFloat64) {
    LaunchKernel<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt32 || dtype_ == kNumberTypeInt16 || dtype_ == kNumberTypeInt64) {
    LaunchKernel<int>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "Data type is " << TypeIdLabel(dtype_) << "is not support.";
  }
  return true;
}

template <typename T>
void ArithmeticSelfCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                           const std::vector<AddressPtr> &outputs) {
  T *input = reinterpret_cast<T *>(inputs[0]->addr);
  T *output = reinterpret_cast<T *>(outputs[0]->addr);
  size_t lens = outputs[0]->size > 0 ? static_cast<size_t>(outputs[0]->size / sizeof(T)) : 1;

  auto max_thread_num = std::thread::hardware_concurrency();
  size_t thread_num = lens < 128 * max_thread_num ? std::ceil(lens / 128.0) : max_thread_num;
  MS_LOG(INFO) << "Lens=" << lens << "; use thread_num=" << thread_num << "; max_thread_num: " << max_thread_num;
  std::vector<std::thread> threads;
  threads.reserve(thread_num);
  size_t start = 0;
  size_t once_compute_size = (lens + thread_num - 1) / thread_num;
  while (start < lens) {
    size_t end = (start + once_compute_size) > lens ? lens : (start + once_compute_size);
    if (operate_type_ == SQUARE) {
      threads.emplace_back(std::thread(Square<T>, input, output, start, end));
    } else if (operate_type_ == NEG) {
      threads.emplace_back(std::thread(Neg<T>, input, output, start, end));
    }
    start += once_compute_size;
  }
  for (size_t i = 0; i < threads.size(); ++i) {
    threads[i].join();
  }
}
}  // namespace kernel
}  // namespace mindspore

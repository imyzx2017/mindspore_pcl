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
#include "backend/kernel_compiler/cpu/debug_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
void DebugCPUKernel::InitKernel(const CNodePtr &kernel_node) { MS_EXCEPTION_IF_NULL(kernel_node); }

bool DebugCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                            const std::vector<kernel::AddressPtr> & /*workspace*/,
                            const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.size() < 1 || outputs.empty()) {
    MS_LOG(EXCEPTION) << " input or output empty!";
  }
  auto val = reinterpret_cast<float *>(inputs[0]->addr);
  MS_LOG(DEBUG) << " launch DebugCountCPUKernel val " << *val;

  auto output = reinterpret_cast<int *>(outputs[0]->addr);
  size_t elem_num = inputs[0]->size / sizeof(int);
  for (size_t i = 0; i < elem_num; i++) {
    output[i] = val[i];
  }

  return true;
}
}  // namespace kernel
}  // namespace mindspore

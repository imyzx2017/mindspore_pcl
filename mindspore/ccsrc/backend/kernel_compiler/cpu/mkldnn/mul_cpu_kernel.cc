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
#include "backend/kernel_compiler/cpu/mkldnn/mul_cpu_kernel.h"
#include "backend/kernel_compiler/cpu/mkldnn/mkl_kernel_engine.h"
#include "runtime/device/cpu/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
void MulCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::vector<size_t> src0_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  std::vector<size_t> src1_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  std::vector<size_t> dst_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  if (src1_shape.size() != src0_shape.size()) {
    if (src0_shape.size() == 0) {
      need_swap_ = true;
      for (size_t i = 0; i < src1_shape.size(); ++i) {
        src0_shape.emplace_back(1);
      }
    } else if (src1_shape.size() == 0) {
      for (size_t i = 0; i < src0_shape.size(); ++i) {
        src1_shape.emplace_back(1);
      }
    } else {
      MS_LOG(EXCEPTION) << "Invalid broadcast! " << src0_shape << " vs " << src1_shape;
    }
  } else {
    bool visit_src0 = false;
    bool visit_src1 = false;
    for (size_t i = 0; i < src0_shape.size(); ++i) {
      if (src0_shape[i] != src1_shape[i]) {
        if (src0_shape[i] == 1 && !visit_src1) {
          need_swap_ = true;
          visit_src0 = true;
        } else if (src1_shape[i] == 1 && !visit_src0) {
          need_swap_ = false;
          visit_src1 = true;
        } else {
          MS_LOG(EXCEPTION) << "Invalid broadcast! " << src0_shape << " vs " << src1_shape;
        }
      }
    }
  }
  dnnl::memory::desc src0_desc;
  dnnl::memory::desc src1_desc;
  if (need_swap_) {
    src0_desc = GetDefaultMemDesc(src1_shape);
    src1_desc = GetDefaultMemDesc(src0_shape);
  } else {
    src0_desc = GetDefaultMemDesc(src0_shape);
    src1_desc = GetDefaultMemDesc(src1_shape);
  }
  dnnl::memory::desc dst_desc = GetDefaultMemDesc(dst_shape);
  dnnl::binary::desc desc = dnnl::binary::desc(dnnl::algorithm::binary_mul, src0_desc, src1_desc, dst_desc);
  auto prim_desc = dnnl::binary::primitive_desc(desc, MKLKernelEngine::Get().engine());
  primitive_ = std::make_shared<dnnl::binary>(prim_desc);

  AddArgument(DNNL_ARG_SRC_0, src0_desc);
  AddArgument(DNNL_ARG_SRC_1, src1_desc);
  AddArgument(DNNL_ARG_DST, dst_desc);
}

bool MulCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                          const std::vector<kernel::AddressPtr> & /*workspace*/,
                          const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.size() < 2 || outputs.empty()) {
    MS_LOG(EXCEPTION) << "mul error input output size!";
  }
  if (need_swap_) {
    SetArgumentHandle(DNNL_ARG_SRC_0, inputs[1]->addr);
    SetArgumentHandle(DNNL_ARG_SRC_1, inputs[0]->addr);
  } else {
    SetArgumentHandle(DNNL_ARG_SRC_0, inputs[0]->addr);
    SetArgumentHandle(DNNL_ARG_SRC_1, inputs[1]->addr);
  }
  SetArgumentHandle(DNNL_ARG_DST, outputs[0]->addr);
  ExecutePrimitive();
  return true;
}
}  // namespace kernel
}  // namespace mindspore

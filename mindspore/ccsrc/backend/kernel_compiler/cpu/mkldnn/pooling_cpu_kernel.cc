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
#include "backend/kernel_compiler/cpu/mkldnn/pooling_cpu_kernel.h"
#include <string>
#include <algorithm>
#include "utils/ms_utils.h"
#include "backend/kernel_compiler/cpu/mkldnn/mkl_kernel_engine.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
void PoolingCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::vector<size_t> src_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  std::vector<size_t> dst_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  dnnl::memory::desc src_desc = GetDefaultMemDesc(src_shape);
  dnnl::memory::desc dst_desc = GetDefaultMemDesc(dst_shape);
  std::vector<int> origin_kernel_sizes;
  std::vector<int> strides;
  std::vector<int64_t> kernel_sizes_me = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, KSIZE);
  std::vector<int64_t> strides_me = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, STRIDES);
  (void)std::transform(kernel_sizes_me.begin(), kernel_sizes_me.end(), std::back_inserter(origin_kernel_sizes),
                       [](const int64_t &value) { return static_cast<int>(value); });
  (void)std::transform(strides_me.begin(), strides_me.end(), std::back_inserter(strides),
                       [](const int64_t &value) { return static_cast<int>(value); });
  if (origin_kernel_sizes.size() != 4 || strides.size() != 4) {
    MS_LOG(EXCEPTION) << "invalid kernel size " << origin_kernel_sizes.size() << " or stride size " << strides.size();
  }
  dnnl::memory::dims strides_dims{strides[2], strides[3]};
  dnnl::memory::dims kernels_dims{origin_kernel_sizes[2], origin_kernel_sizes[3]};
  const std::string pad_mode = AnfAlgo::GetNodeAttr<std::string>(kernel_node, PADDING);
  std::vector<int> int_padding_l;
  std::vector<int> int_padding_r;
  std::vector<size_t> kernel_size({IntToSize(origin_kernel_sizes[2]), IntToSize(origin_kernel_sizes[3])});
  GetPadding(kernel_node, pad_mode, src_shape, kernel_size, strides[3], &int_padding_l, &int_padding_r);
  if (int_padding_l.size() != 2 || int_padding_r.size() != 2) {
    MS_LOG(EXCEPTION) << "pooling get padding failed";
  }
  dnnl::memory::dims padding_l{int_padding_l[0], int_padding_l[1]};
  dnnl::memory::dims padding_r{int_padding_r[0], int_padding_r[1]};
  dnnl::pooling_forward::desc desc =
    dnnl::pooling_forward::desc(dnnl::prop_kind::forward_training, dnnl::algorithm::pooling_max, src_desc, dst_desc,
                                strides_dims, kernels_dims, padding_l, padding_r);
  auto prim_desc = dnnl::pooling_forward::primitive_desc(desc, MKLKernelEngine::Get().engine());
  primitive_ = std::make_shared<dnnl::pooling_forward>(prim_desc);
  AddArgument(DNNL_ARG_SRC, src_desc);
  AddArgument(DNNL_ARG_DST, dst_desc);
  AddArgument(DNNL_ARG_WORKSPACE, prim_desc.workspace_desc());
}

bool PoolingCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                              const std::vector<kernel::AddressPtr> & /*workspace*/,
                              const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(EXCEPTION) << "error input output size!";
  }
  SetArgumentHandle(DNNL_ARG_SRC, inputs[0]->addr);
  SetArgumentHandle(DNNL_ARG_DST, outputs[0]->addr);
  ExecutePrimitive();
  return true;
}
}  // namespace kernel
}  // namespace mindspore

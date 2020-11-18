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
#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_CPU_CPU_KERNEL_RUNTIME_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_CPU_CPU_KERNEL_RUNTIME_H_

#include <memory>
#include <vector>
#include <string>
#include <map>
#include <set>
#include "runtime/device/kernel_runtime.h"
#include "backend/session/kernel_graph.h"
#include "backend/session/session_basic.h"
#include "runtime/device/cpu/cpu_resource_manager.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "utils/any.h"
namespace mindspore {
namespace device {
namespace cpu {
class CPUKernelRuntime : public KernelRuntime {
 public:
  CPUKernelRuntime() = default;
  ~CPUKernelRuntime() override = default;

  bool Init() override { return true; }
  bool Run(session::KernelGraph *graph, bool is_task_sink) override;
  void AssignKernelAddress(session::KernelGraph *kernel_graph);
  void CreateOutputTensors(session::KernelGraph *kernel_graph, const std::vector<tensor::TensorPtr> &inputs,
                           VectorRef *outputs);
  void BindInputOutput(session::KernelGraph *kernel_graph, const std::vector<tensor::TensorPtr> &inputs,
                       VectorRef *outputs);
  void IncreaseSummaryRefCount(const session::NamedSummaryOutputs &summary_outputs);
  void DecreaseSummaryRefCount(const session::NamedSummaryOutputs &summary_outputs);
  bool GenDynamicKernel(const session::KernelGraph *graph) override { return true; }
  bool RunDynamicKernelAsync(const session::KernelGraph *graph) override { return true; }

 protected:
  bool SyncStream() override { return true; };
  DeviceAddressPtr CreateDeviceAddress(void *device_ptr, size_t device_size, const string &format,
                                       TypeId type_id) override;

 private:
  tensor::TensorPtr CreatTensorForOutput(session::KernelGraph *kernel_graph, const CNodePtr &node, size_t index);
  BaseRef CreatTensorForOutput(session::KernelGraph *kernel_graph, const session::KernelWithIndex &kernel_with_index);
  void BindInputTensorAddressPtr(const session::KernelGraph &graph, const std::vector<tensor::TensorPtr> &inputs);
  void BindOutputTensorAddressPtr(const VectorRef *outputs);
  void AssignValueNodeAddress(session::KernelGraph *kernel_graph);
  void AssignInputNodeAddress(const session::KernelGraph *kernel_graph);
  void AssignKernelOutputAddress(const session::KernelGraph *kernel_graph);
  void AddRuntimeAddress(DeviceAddress *address, std::vector<kernel::AddressPtr> *input_list);
  CPUResourceManager resource_manager_;
  std::set<DeviceAddressPtr> bound_addresses_;
  std::map<AnfNodePtr, tensor::TensorPtr> input_param_tensor_map_;
};
}  // namespace cpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_CPU_CPU_KERNEL_RUNTIME_H_

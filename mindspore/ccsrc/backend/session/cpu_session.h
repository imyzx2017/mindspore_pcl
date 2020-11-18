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
#ifndef MINDSPORE_CCSRC_BACKEND_SESSION_CPU_SESSION_H
#define MINDSPORE_CCSRC_BACKEND_SESSION_CPU_SESSION_H
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "backend/session/session_basic.h"
#include "backend/session/kernel_graph.h"
#include "runtime/device/cpu/cpu_kernel_runtime.h"
#include "backend/session/session_factory.h"
namespace mindspore {
namespace session {
class CPUSession : public SessionBasic {
 public:
  CPUSession() = default;
  ~CPUSession() override = default;
  void Init(uint32_t device_id) override { InitDevice(kCPUDevice, device_id); }

 protected:
  void CreateOutputTensors(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &input_tensors, VectorRef *,
                           std::map<tensor::TensorPtr, session::KernelWithIndex> *tensor_to_node) override;
  GraphId CompileGraphImpl(const AnfNodePtrList &lst, const AnfNodePtrList &outputs) override;
  void RunGraphImpl(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs, VectorRef *outputs) override;
  ParameterPtr CreateNewParameterFromParameter(const AnfNodePtr &anf, KernelGraph *graph) override;
  void Optimize(const std::shared_ptr<KernelGraph> &kernel_graph);

 private:
  void SetKernelInfo(const KernelGraph *kernel_graph);
  void BuildKernel(const KernelGraph *kernel_graph);
  device::cpu::CPUKernelRuntime runtime_;
};
MS_REG_SESSION(kCPUDevice, CPUSession);
}  // namespace session
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_SESSION_CPU_SESSION_H

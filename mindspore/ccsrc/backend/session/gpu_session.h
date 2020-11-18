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
#ifndef MINDSPORE_CCSRC_BACKEND_SESSION_GPU_SESSION_H
#define MINDSPORE_CCSRC_BACKEND_SESSION_GPU_SESSION_H

#include <vector>
#include <memory>
#include <algorithm>
#include "backend/session/session_basic.h"
#include "backend/session/kernel_graph.h"
#include "backend/session/session_factory.h"
using KernelGraph = mindspore::session::KernelGraph;

namespace mindspore {
namespace session {
namespace gpu {
class GPUSession : public SessionBasic {
 public:
  GPUSession() = default;
  ~GPUSession() override = default;
  void Init(uint32_t device_id) override;

 protected:
  GraphId CompileGraphImpl(const AnfNodePtrList &lst, const AnfNodePtrList &outputs) override;
  void RunGraphImpl(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs, VectorRef *outputs) override;
  void BuildOpImpl(const OpRunInfo &op_run_info, const GraphInfo &graph_info,
                   const std::vector<tensor::TensorPtr> &input_tensors,
                   const std::vector<int64_t> &tensors_mask) override;
  void RunOpImpl(const OpRunInfo &op_run_info, const GraphInfo &graph_info,
                 const std::vector<tensor::TensorPtr> &input_tensors, VectorRef *outputs) override;

 private:
  void SelectKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const;

  void StartKernelRT() const;

  void Optimize(const std::shared_ptr<KernelGraph> &kernel_graph);

  void HardwareOptimize(const std::shared_ptr<KernelGraph> &kernel_graph);

  void RunOpHardwareOptimize(const std::shared_ptr<KernelGraph> &kernel_graph);

  void GraphKernelOptimize(const std::shared_ptr<KernelGraph> &kernel_graph);

  void AssignStream(const std::shared_ptr<KernelGraph> &kernel_graph);

  void BuildKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const;

  void AllocateMemory(KernelGraph *kernel_graph) const;

  void RunOpAllocateMemory(const ValuePtr &pre_output_value, const std::vector<tensor::TensorPtr> &input_tensors,
                           KernelGraph *kernel_graph) const;

  void RunOpClearMemory(KernelGraph *kernel_graph) const;

  void LoadInputData(const std::shared_ptr<KernelGraph> &kernel_graph,
                     const std::vector<tensor::TensorPtr> &inputs_const) const override;

  void Execute(const std::shared_ptr<KernelGraph> &kernel_graph) const;

  void Dump(const std::shared_ptr<KernelGraph> &kernel_graph) const;

  bool DumpDataEnabledIteration() const;

  void PreIterationDbg(const std::shared_ptr<KernelGraph> &kernel_graph) const;

  void PostIterationDbg(const std::shared_ptr<KernelGraph> &kernel_graph) const;

  void PreLoadTensor(const std::shared_ptr<KernelGraph> &kernel_graph) const;

  void PostLoadTensor(const std::shared_ptr<KernelGraph> &kernel_graph) const;
};
using GPUSessionPtr = std::shared_ptr<GPUSession>;
MS_REG_SESSION(kGPUDevice, GPUSession);
}  // namespace gpu
}  // namespace session
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_SESSION_GPU_SESSION_H

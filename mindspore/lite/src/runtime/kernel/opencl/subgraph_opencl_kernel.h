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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_SUBGRAPH_OPENCL_KENEL_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_SUBGRAPH_OPENCL_KENEL_H_

#include <set>
#include <vector>
#include "src/runtime/kernel/opencl/opencl_kernel.h"
#include "src/runtime/opencl/opencl_allocator.h"
#include "src/runtime/opencl/opencl_executor.h"
#include "src/sub_graph_kernel.h"

namespace mindspore::kernel {
struct SubGraphOpenCLParameter {
  OpParameter op_parameter;
  int input_size;
  int output_size;
};

class SubGraphOpenCLKernel : public SubGraphKernel {
 public:
  SubGraphOpenCLKernel(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                       const std::vector<kernel::LiteKernel *> &inKernels,
                       const std::vector<kernel::LiteKernel *> &outKernels,
                       const std::vector<kernel::LiteKernel *> &nodes, const lite::InnerContext *ctx = nullptr)
      : SubGraphKernel(inputs, outputs, inKernels, outKernels, nodes, ctx) {
    ocl_runtime_ = ocl_runtime_wrap_.GetInstance();
    subgraph_type_ = kGpuSubGraph;
    this->name_ = "GpuSubGraph";
    this->executor_ = new lite::opencl::OpenCLExecutor();
    nodes_set_.insert(nodes.begin(), nodes.end());
  }
  ~SubGraphOpenCLKernel() override;

  int PreProcess() override { return mindspore::lite::RET_OK; }
  int PostProcess() override { return mindspore::lite::RET_OK; }
  int Prepare() override;
  int Init() override;
  int InferShape();
  int ReSize() override;
  int Run() override;
  int Run(const KernelCallBack &before, const KernelCallBack &after) override { return this->Run(); };

 private:
  int UnInit();
  int UpdateTensorDataType();
  int MallocTensorWithReuse();
  int ReplaceOutTensorAndKernelToNull(const std::vector<lite::Tensor *> &in_tensors,
                                      const std::vector<std::vector<kernel::LiteKernel *>> &in_kernels,
                                      lite::opencl::MemType mem_type);
  int ReplaceOutTensorAndKernelToConvert(const lite::Tensor *in_tensor,
                                         const std::vector<kernel::LiteKernel *> &in_kernels, lite::Tensor *new_tensor,
                                         kernel::LiteKernel *in_convert_op, lite::opencl::MemType mem_type);
  int GetInOutNodes();
  int GenToFormatOp(const std::vector<lite::Tensor *> &in_tensors,
                    const std::vector<std::vector<kernel::LiteKernel *>> &in_kernels,
                    std::vector<lite::Tensor *> *out_tensors, std::vector<OpenCLToFormatParameter *> *out_parameters,
                    std::vector<LiteKernel *> *out_convert_ops, lite::opencl::MemType mem_type);
  int GetKernelFromToTensor(const std::vector<lite::Tensor *> &in_tensors,
                            const std::vector<kernel::LiteKernel *> &in_kernels,
                            std::vector<std::vector<kernel::LiteKernel *>> *out_kernels, bool is_from);
  lite::opencl::OpenCLAllocator *allocator_{nullptr};
  std::vector<lite::Tensor *> in_convert_tensors_;
  std::vector<lite::Tensor *> out_convert_tensors_;
  std::vector<OpenCLToFormatParameter *> in_parameters_;
  std::vector<OpenCLToFormatParameter *> out_parameters_;
  std::vector<LiteKernel *> in_convert_ops_;
  std::vector<LiteKernel *> out_convert_ops_;
  std::vector<LiteKernel *> in_nodes_;
  std::vector<LiteKernel *> out_nodes_;
  std::set<LiteKernel *> nodes_set_;
  lite::opencl::OpenCLRuntimeWrapper ocl_runtime_wrap_;
  lite::opencl::OpenCLRuntime *ocl_runtime_{nullptr};
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_KERNEL_SUBGRAPH_OPENCL_KERNEL_H_

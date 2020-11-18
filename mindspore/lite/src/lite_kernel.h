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

#ifndef MINDSPORE_LITE_SRC_LITE_KERNEL_H_
#define MINDSPORE_LITE_SRC_LITE_KERNEL_H_
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include "src/ops/primitive_c.h"
#include "src/common/utils.h"
#ifdef ENABLE_ARM
#include <arm_neon.h>
#endif
#include "nnacl/op_base.h"
#include "src/inner_context.h"
#include "src/tensor.h"
#include "include/errorcode.h"

static constexpr int kPerTensor = 1;
static constexpr size_t kPerBatch = 3;

namespace mindspore::kernel {
enum KERNEL_ARCH { kCPU, kGPU, kAPU, kNPU, kKernelArch_MIN = kCPU, kKernelArch_MAX = kNPU };
struct KernelKey {
  KERNEL_ARCH arch;
  TypeId data_type;
  schema::PrimitiveType type;

  bool operator<(const KernelKey &dst) const {
    if (arch != dst.arch) {
      return arch < dst.arch;
    } else if (data_type != dst.data_type) {
      return data_type < dst.data_type;
    } else {
      return type < dst.type;
    }
  }
};

enum SubGraphType { kNotSubGraph = 0, kCpuFP32SubGraph, kCpuFP16SubGraph, kGpuSubGraph, kNpuSubGraph, kApuSubGraph };

class LiteKernel {
 public:
  LiteKernel() = default;
  // parameter should be deleted or freed by caller, and should be deleted or freed after LiteKernel is deleted
  LiteKernel(OpParameter *parameter, std::vector<lite::Tensor *> in_tensors, std::vector<lite::Tensor *> out_tensors,
             const lite::InnerContext *ctx, const mindspore::lite::PrimitiveC *primitive)
      : op_parameter_(parameter),
        in_tensors_(std::move(in_tensors)),
        out_tensors_(std::move(out_tensors)),
        primitive_(primitive),
        context_(ctx) {
    if (op_parameter_ != nullptr && ctx != nullptr) {
      op_parameter_->thread_num_ = ctx->thread_num_;
    }
    this->in_kernels_.clear();
    this->out_kernels_.clear();
  }

  virtual ~LiteKernel() {
    if (op_parameter_ != nullptr) {
      free(op_parameter_);
      op_parameter_ = nullptr;
    }
  }

  // called while compiling graph
  virtual int Prepare() { return mindspore::lite::RET_OK; }
  // called before Run
  virtual int PreProcess();

  virtual int Run() { return mindspore::lite::RET_ERROR; }

  virtual int Run(const KernelCallBack &before, const KernelCallBack &after);
  // called after Run
  virtual int PostProcess() { return FreeWorkTensor(); }

  virtual int ReSize() { return mindspore::lite::RET_ERROR; }

  virtual int Init() { return mindspore::lite::RET_ERROR; }

  std::string name() const { return this->name_; }

  virtual int Train() {
    this->train_mode_ = true;
    return mindspore::lite::RET_OK;
  }

  virtual bool IsTrain() const { return this->train_mode_; }

  virtual int Eval() {
    this->train_mode_ = false;
    return mindspore::lite::RET_OK;
  }

  virtual bool IsEval() const { return !this->train_mode_; }

  virtual void SetTrainable(bool trainable = true) { this->trainable_ = trainable; }

  virtual bool IsTrainable() const { return this->trainable_; }

  void set_name(const std::string &name) { this->name_ = name; }

  void set_is_model_output(bool is_model_output) { this->is_model_output_ = is_model_output; }

  bool is_model_output() const { return this->is_model_output_; }

  schema::PrimitiveType Type() const {
    return (this->op_parameter_ != nullptr) ? schema::PrimitiveType(this->op_parameter_->type_)
                                            : schema::PrimitiveType_NONE;
  }

  std::string type_str() const { return schema::EnumNamePrimitiveType(this->Type()); }

  void set_in_tensors(const std::vector<lite::Tensor *> &in_tensors) { this->in_tensors_ = in_tensors; }

  void set_out_tensors(const std::vector<lite::Tensor *> &out_tensors) { this->out_tensors_ = out_tensors; }

  const std::vector<lite::Tensor *> &in_tensors() const { return this->in_tensors_; }

  const std::vector<lite::Tensor *> &out_tensors() const { return this->out_tensors_; }

  void AddInKernel(LiteKernel *kernel) {
    if (!lite::IsContain(this->in_kernels_, kernel)) {
      this->in_kernels_.emplace_back(kernel);
    }
  }

  void AddOutKernel(LiteKernel *kernel) {
    if (!lite::IsContain(this->out_kernels_, kernel)) {
      this->out_kernels_.emplace_back(kernel);
    }
  }

  void SetInKernel(const std::vector<LiteKernel *> &kernel) { this->in_kernels_ = kernel; }

  void SetOutKernel(const std::vector<LiteKernel *> &kernel) { this->out_kernels_ = kernel; }

  const std::vector<LiteKernel *> &in_kernels() const { return this->in_kernels_; }

  const std::vector<LiteKernel *> &out_kernels() const { return this->out_kernels_; }

  void InitOutTensorRefCount();

  int DecOutTensorRefCount();

  int FreeWorkTensor() const;

  KernelKey desc() const { return desc_; }

  void set_desc(const KernelKey kernel_key) { desc_ = kernel_key; }

  const mindspore::lite::PrimitiveC *GetPrimitive() const { return primitive_; }
  void SetWorkspaceSize(size_t value) { workspace_size_ = value; }
  size_t GetWorkspaceSize() { return workspace_size_; }
  static void AllocWorkspace(size_t size);
  static void FreeWorkspace();
  void *GetWorkspace() { return workspace_; }

  SubGraphType subgraph_type() const { return this->subgraph_type_; }

  virtual std::string ToString() const;

 protected:
  bool InferShapeDone() { return !(primitive_ != nullptr && !primitive_->GetInferFlag()); }

  KernelKey desc_{};
  std::string name_;
  OpParameter *op_parameter_ = nullptr;
  // tensor will free in ~lite_session()
  std::vector<lite::Tensor *> in_tensors_;
  std::vector<lite::Tensor *> out_tensors_;
  const mindspore::lite::PrimitiveC *primitive_ = nullptr;
  const lite::InnerContext *context_ = nullptr;
  std::vector<LiteKernel *> in_kernels_;
  std::vector<LiteKernel *> out_kernels_;
  bool train_mode_ = false;
  bool trainable_ = false;  // paramaters of this Kernel are trained in Train Session
  bool is_model_output_ = false;
  size_t workspace_size_ = 0;
  static void *workspace_;
  SubGraphType subgraph_type_ = kNotSubGraph;
};

typedef LiteKernel *(*KernelCreator)(const std::vector<lite::Tensor *> &inputs,
                                     const std::vector<lite::Tensor *> &outputs, OpParameter *parameter,
                                     const lite::InnerContext *ctx, const KernelKey &desc,
                                     const mindspore::lite::PrimitiveC *primitive);

class LiteKernelUtil {
 public:
  static void InitIOKernels(std::vector<kernel::LiteKernel *> &kernels);

  static std::vector<kernel::LiteKernel *> SubgraphInputKernels(const std::vector<kernel::LiteKernel *> &kernels);

  static std::vector<kernel::LiteKernel *> SubgraphOutputKernels(const std::vector<kernel::LiteKernel *> &kernels);

  static std::vector<lite::Tensor *> SubgraphInputTensors(const std::vector<kernel::LiteKernel *> &kernels);

  static std::vector<lite::Tensor *> SubgraphOutputTensors(const std::vector<kernel::LiteKernel *> &kernels);

  static int TopologicalSortKernels(std::vector<kernel::LiteKernel *> *kernels);

  static void InitTensorRefCount(std::vector<kernel::LiteKernel *> &kernels);

  static int SetInput(LiteKernel &kernelMod, std::vector<lite::Tensor *> inputs);
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_LITE_KERNEL_H_

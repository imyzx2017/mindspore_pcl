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

#include "src/lite_session.h"
#include <vector>
#include <utility>
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "src/scheduler.h"
#include "src/runtime/allocator.h"
#include "src/executor.h"
#include "src/common/utils.h"
#include "src/common/graph_util.h"
#include "src/kernel_registry.h"
#include "src/model_common.h"
#include "mindspore/lite/src/runtime/kernel/arm/base/dequant.h"

namespace mindspore {
namespace lite {
static std::vector<schema::PrimitiveType> packed_op = {
  schema::PrimitiveType_Conv2D, schema::PrimitiveType_DeConv2D, schema::PrimitiveType_DepthwiseConv2D,
  schema::PrimitiveType_DeDepthwiseConv2D, schema::PrimitiveType_MatMul};

// this method will not check whether tensor_idx is a weight tensor index, caller should ensure this.
static bool WeightTensorNeedCopy(const lite::Model *model, const uint32_t tensor_idx) {
#ifdef SUPPORT_TRAIN
  return false;
#endif

  MS_ASSERT(model != nullptr);
  auto post_node_idxes = GetLinkedPostNodeIdx(model, tensor_idx);
  return std::none_of(post_node_idxes.begin(), post_node_idxes.end(), [&](const size_t &post_node_idx) {
    auto node = model->all_nodes_[post_node_idx];
    MS_ASSERT(node != nullptr);
    return IsContain(packed_op, static_cast<schema::PrimitiveType>(node->primitive_->Type()));
  });
}

LiteSession::LiteSession() { this->is_running_.store(false); }

int LiteSession::ConvertTensors(const lite::Model *model) {
  MS_ASSERT(model != nullptr);
  copyed_tensor_idxes_.clear();
  uint32_t tensor_count = model->all_tensors_.size();
  for (uint32_t i = 0; i < tensor_count; ++i) {
    auto *srcTensor = model->all_tensors_[i];
    if (srcTensor == nullptr) {
      MS_LOG(ERROR) << i << "th tensor in model is nullptr";
      return RET_NULL_PTR;
    }
    auto src_category = TensorCategory(srcTensor);
    std::vector<int> shape;
    if (srcTensor->dims() == nullptr) {
      MS_LOG(DEBUG) << "Dims of " << i << "th tensor is nullptr";
    } else {
      if (src_category == Tensor::Category::CONST_TENSOR) {
        if (srcTensor->dataType() == kObjectTypeString && srcTensor->data() != nullptr) {
          shape.push_back(srcTensor->data()->size());
        } else {
          for (size_t j = 0; j < srcTensor->dims()->size(); j++) {
            shape.push_back(srcTensor->dims()->data()[j]);
          }
        }
      }
    }
    int dataType = srcTensor->dataType();
    auto *dstTensor = new (std::nothrow) Tensor(TypeId(dataType), shape, srcTensor->format(), src_category);
    if (dstTensor == nullptr) {
      MS_LOG(ERROR) << "new " << i << "th tensor failed";
      return RET_NULL_PTR;
    }
    if ((src_category == Tensor::Category::CONST_TENSOR || src_category == Tensor::Category::CONST_SCALAR) &&
        srcTensor->data() != nullptr && srcTensor->data()->size() > 0) {
      MS_ASSERT(dstTensor->Size() == srcTensor->data()->size());
      if (WeightTensorNeedCopy(model, i)) {
        auto dst_data = dstTensor->MutableData();
        if (dst_data == nullptr) {
          MS_LOG(ERROR) << "MutableData from " << i << "th tensor is nullptr";
          delete dstTensor;
          return RET_ERROR;
        }
        memcpy(dst_data, srcTensor->data()->data(), dstTensor->Size());
        copyed_tensor_idxes_.emplace_back(i);
      } else {
        int pack_size = srcTensor->data()->size();
        int org_size = dstTensor->Size();
        if (pack_size != org_size && (dataType == kNumberTypeInt8 || dataType == kNumberTypeInt16)) {
          auto ret = dstTensor->MallocData();
          if (ret != RET_OK) {
            MS_LOG(ERROR) << "Malloc data for " << i << "tensor failed ";
            delete dstTensor;
            return RET_ERROR;
          }
          kernel::DequantUtil::UnPackToInt(srcTensor, dstTensor->MutableData());
        } else {
          dstTensor->set_data(const_cast<unsigned char *>(srcTensor->data()->data()));
        }
      }
    }
    auto quant_params = srcTensor->quantParams();
    if (quant_params != nullptr) {
      for (size_t j = 0; j < quant_params->size(); j++) {
        QuantArg quant_arg{};
        quant_arg.bitNum = quant_params->Get(j)->numBits();
        quant_arg.scale = quant_params->Get(j)->scale();
        quant_arg.zeroPoint = quant_params->Get(j)->zeroPoint();
        quant_arg.var_corr = quant_params->Get(j)->varCorr();
        quant_arg.mean_corr = quant_params->Get(j)->meanCorr();
        quant_arg.inited = quant_params->Get(j)->inited();
        dstTensor->AddQuantParam(quant_arg);
      }
    }
    auto quant_clusters = srcTensor->quantClusters();
    if (quant_clusters != nullptr) {
      std::vector<float> clusters;
      for (size_t j = 0; j < quant_clusters->size(); j++) {
        clusters.push_back(quant_clusters->Get(j));
      }
      dstTensor->SetQuantClusters(clusters);
    }
    this->tensors_.emplace_back(dstTensor);
  }

  return RET_OK;
}

void LiteSession::InitGraphInputTensors(const lite::Model *model) {
  MS_ASSERT(model != nullptr);
  MS_ASSERT(!(model->sub_graphs_.empty()));
  auto graph_in_size = model->sub_graphs_.front()->input_indices_.size();
  for (size_t i = 0; i < graph_in_size; ++i) {
    auto in_tensor_idx = model->sub_graphs_.front()->input_indices_[i];
    MS_ASSERT(in_tensor_idx < this->tensors_.size());
    auto *in_tensor = this->tensors_.at(in_tensor_idx);
    MS_ASSERT(in_tensor != nullptr);
    this->inputs_.emplace_back(in_tensor);
  }
}

void LiteSession::InitGraphInputMSTensors() {
  MS_ASSERT(this->input_vec_.empty());
  for (auto &input_tensor : this->inputs_) {
    MS_ASSERT(input_tensor != nullptr);
    this->input_vec_.emplace_back(input_tensor);
  }
}

void LiteSession::InitGraphOutputTensors(const lite::Model *model) {
  MS_ASSERT(model != nullptr);
  MS_ASSERT(this->outputs_.empty());
  auto graph_out_size = model->sub_graphs_.front()->output_indices_.size();
  for (size_t i = 0; i < graph_out_size; ++i) {
    auto out_tensor_idx = model->sub_graphs_.front()->output_indices_[i];
    MS_ASSERT(out_tensor_idx < this->tensors_.size());
    auto *out_tensor = this->tensors_.at(out_tensor_idx);
    MS_ASSERT(out_tensor != nullptr);
    this->outputs_.emplace_back(out_tensor);
  }
}

void LiteSession::InitGraphInputMap(const lite::Model *model) {
  MS_ASSERT(model != nullptr);
  MS_ASSERT(this->input_map_.empty());
  auto graph_input_node_indexes = GetGraphInputNodes(model);
  auto graph_in_size = model->sub_graphs_.front()->input_indices_.size();
  for (auto in_node_index : graph_input_node_indexes) {
    auto in_node = model->all_nodes_[in_node_index];
    MS_ASSERT(in_node != nullptr);
    MS_ASSERT(this->input_map_.find(in_node->name_) == this->input_map_.end());
    auto in_size = in_node->input_indices_.size();
    for (size_t i = 0; i < in_size; ++i) {
      auto in_tensor_index = size_t(in_node->input_indices_[i]);
      bool is_graph_input = false;
      for (size_t j = 0; j < graph_in_size; ++j) {
        if (in_tensor_index == model->sub_graphs_.front()->input_indices_[j]) {
          is_graph_input = true;
          break;
        }
      }
      if (!is_graph_input) {
        continue;
      }
      MS_ASSERT(in_tensor_index < this->tensors_.size());
      auto *in_tensor = this->tensors_.at(in_tensor_index);
      if (in_tensor == nullptr) {
        MS_LOG(ERROR) << "in_tensor is null!";
        return;
      }
      auto tensor_name = in_node->name_ + std::to_string(i);
      this->input_map_[tensor_name] = in_tensor;
    }
  }
}

void LiteSession::InitGraphOutputNodeMap(const lite::Model *model) {
  MS_ASSERT(model != nullptr);
  MS_ASSERT(!(model->sub_graphs_.empty()));
  auto graph_output_node_indexes = GetGraphOutputNodes(model);
  auto graph_out_size = model->sub_graphs_.front()->output_indices_.size();
  for (auto out_node_index : graph_output_node_indexes) {
    auto out_node = model->all_nodes_[out_node_index];
    MS_ASSERT(out_node != nullptr);
    auto out_size = out_node->output_indices_.size();
    for (size_t i = 0; i < out_size; ++i) {
      auto out_tensor_index = out_node->output_indices_[i];
      bool is_graph_output = false;
      for (size_t j = 0; j < graph_out_size; ++j) {
        if (out_tensor_index == model->sub_graphs_.front()->output_indices_[j]) {
          is_graph_output = true;
          break;
        }
      }
      if (!is_graph_output) {
        continue;
      }
      MS_ASSERT(out_tensor_index < this->tensors_.size());
      auto *out_tensor = this->tensors_.at(out_tensor_index);
      if (out_tensor == nullptr) {
        MS_LOG(ERROR) << "out_tensor is null!";
        return;
      }
      this->output_node_map_[out_node->name_].emplace_back(out_tensor);
    }
  }
}

void LiteSession::InitGraphOutputTensorNames(const lite::Model *model) {
  MS_ASSERT(model != nullptr);
  MS_ASSERT(this->output_tensor_names_.empty());
  auto out_size = model->sub_graphs_.front()->output_indices_.size();
  for (size_t i = 0; i < out_size; ++i) {
    this->output_tensor_names_.emplace_back(std::to_string(model->sub_graphs_.front()->output_indices_[i]));
  }
}

void LiteSession::InitGraphOutputTensorMap(const lite::Model *model) {
  MS_ASSERT(model != nullptr);
  MS_ASSERT(this->output_tensor_map_.empty());
  auto graph_out_size = model->sub_graphs_.front()->output_indices_.size();
  for (size_t i = 0; i < graph_out_size; ++i) {
    size_t graph_out_index = model->sub_graphs_.front()->output_indices_[i];
    MS_ASSERT(graph_out_index < this->tensors_.size());
    auto *out_tensor = this->tensors_.at(graph_out_index);
    if (out_tensor == nullptr) {
      MS_LOG(ERROR) << "out_tensor is null!";
      return;
    }
    this->output_tensor_map_.insert(std::make_pair(std::to_string(graph_out_index), out_tensor));
  }
}

void LiteSession::InitGraphInOutTensors(const lite::Model *model) {
  InitGraphInputTensors(model);
  InitGraphInputMSTensors();
  InitGraphOutputTensors(model);
  InitGraphInputMap(model);
  InitGraphOutputNodeMap(model);
  InitGraphOutputTensorNames(model);
  InitGraphOutputTensorMap(model);
}

int LiteSession::CompileGraph(Model *model) {
  bool expected = false;
  if (!is_running_.compare_exchange_strong(expected, true)) {
    MS_LOG(ERROR) << "Not support multi-threading";
    return RET_ERROR;
  }
  // model.MetaGraph ==> kernels
  if (model == nullptr) {
    MS_LOG(ERROR) << "The input model is nullptr.";
    is_running_.store(false);
    return RET_PARAM_INVALID;
  }
  if (model->buf == nullptr) {
    MS_LOG(ERROR) << "The input model buf is nullptr.";
    is_running_.store(false);
    return RET_PARAM_INVALID;
  }

  auto ret = ConvertTensors(model);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvertTensors failed: " << ret;
    is_running_.store(false);
    return ret;
  }

  InitGraphInOutTensors(model);

  // scheduler kernels
  Scheduler scheduler(context_);
  ret = scheduler.Schedule(model, &tensors_, &kernels_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Schedule kernels failed: " << ret;
    is_running_.store(false);
    return ret;
  }
  ret = executor_->Prepare(this->kernels_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare executor failed: " << ret;
    is_running_.store(false);
    return ret;
  }
  ret = PrepareKernels();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare kernels failed: " << ret;
    is_running_.store(false);
    return ret;
  }
  is_running_.store(false);
  return RET_OK;
}

int LiteSession::PrepareKernels() {
  for (auto kernel : this->kernels_) {
    auto ret = kernel->Prepare();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Prepare kernel " << kernel->name() << " failed: " << ret;
      return ret;
    }
  }
  return RET_OK;
}

std::vector<mindspore::tensor::MSTensor *> LiteSession::GetInputs() const { return this->input_vec_; }

int LiteSession::RunGraph(const KernelCallBack &before, const KernelCallBack &after) {
  bool expected = false;
  if (!is_running_.compare_exchange_strong(expected, true)) {
    MS_LOG(ERROR) << "Not support multi-threading";
    return RET_ERROR;
  }
  STATUS ret;
  MS_ASSERT(this->context_);
  if (before == nullptr && after == nullptr) {
    ret = executor_->Run(this->inputs_, this->outputs_, this->kernels_, this->context_->allocator.get());
  } else {
    ret = executor_->Run(this->inputs_, this->outputs_, this->kernels_, this->context_->allocator.get(), before, after);
  }
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "RunGraph failed : " << ret;
  }
  is_running_.store(false);
  return ret;
}

int LiteSession::Init(const Context *context) {
  bool expected = false;
  if (!is_running_.compare_exchange_strong(expected, true)) {
    MS_LOG(ERROR) << "Not support multi-threading";
    return RET_ERROR;
  }

  if (context == nullptr) {
    MS_LOG(ERROR) << "context is nullptr";
    is_running_.store(false);
    return RET_NULL_PTR;
  }

  this->context_ = new (std::nothrow) InnerContext(context);
  if (this->context_ == nullptr) {
    MS_LOG(ERROR) << "New Context failed";
    is_running_.store(false);
    return RET_MEMORY_FAILED;
  }
  auto ret = this->context_->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init Context failed";
    is_running_.store(false);
    return ret;
  }
  ret = KernelRegistry::GetInstance()->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "KernelRegistry Init Failed.";
    is_running_.store(false);
    return ret;
  }
#if SUPPORT_GPU
  if (this->context_->IsGpuEnabled()) {
    auto gpu_device_info = this->context_->GetGpuInfo();
    auto opencl_runtime = ocl_runtime_wrap_.GetInstance();
    opencl_runtime->SetFp16Enable(gpu_device_info.enable_float16_);
    if (opencl_runtime->Init() != RET_OK) {
      this->context_->device_list_ = {{DT_CPU, {gpu_device_info.enable_float16_, MID_CPU}}};
      MS_LOG(WARNING) << "Init OpenCL runtime failed, change to CPU mode.";
    } else {
      MS_LOG(INFO) << "Init OpenCL runtime success.";
    }
  }
#endif
  executor_ = new (std::nothrow) Executor();
  if (nullptr == executor_) {
    MS_LOG(ERROR) << "New Executor failed";
    is_running_.store(false);
    return RET_ERROR;
  }
  is_running_.store(false);
  return RET_OK;
}

void LiteSession::BindThread(bool if_bind) {
  if (this->context_->device_list_.empty()) {
    MS_LOG(ERROR) << "Device list is empty.";
    return;
  }
  if (!this->context_->IsCpuEnabled()) {
    return;
  }
  auto cpu_device_info = this->context_->GetCpuInfo();
  if (cpu_device_info.cpu_bind_mode_ != NO_BIND) {
    MS_ASSERT(this->context_->thread_pool_ != NULL);
    BindThreads(this->context_->thread_pool_, if_bind, cpu_device_info.cpu_bind_mode_);
  }
}

LiteSession::~LiteSession() {
  bool expected = false;
  if (!is_running_.compare_exchange_strong(expected, true)) {
    MS_LOG(ERROR) << "Not support multi-threading";
    return;
  }
  for (size_t i = 0; i < tensors_.size(); i++) {
    auto *tensor = tensors_.at(i);
    MS_ASSERT(tensor != nullptr);
    // data of weight tensor of node in packed_op can not be to free, we will free weight data when freeing meta_graph
    if (tensor->IsConst() && !IsContain(this->inputs_, tensor) && !IsContain(copyed_tensor_idxes_, i)) {
      tensor->set_data(nullptr);
    }
    delete tensor;
  }
  // Tensor * in input_map output_map are freed in tensors
  input_map_.clear();
  output_node_map_.clear();
  output_tensor_map_.clear();
  input_vec_.clear();
  for (auto *kernel : kernels_) {
    delete kernel;
  }
  delete this->context_;
  delete this->executor_;
  this->executor_ = nullptr;
  is_running_.store(false);
}

mindspore::tensor::MSTensor *LiteSession::GetInputsByTensorName(const std::string &name) const {
  auto ret = input_map_.find(name);
  if (ret == input_map_.end()) {
    MS_LOG(WARNING) << "Tensor  " << name << " is not exist";
    return nullptr;
  }
  return ret->second;
}

std::vector<mindspore::tensor::MSTensor *> LiteSession::GetOutputsByNodeName(const std::string &node_name) const {
  auto ret = output_node_map_.find(node_name);
  if (ret == output_node_map_.end()) {
    MS_LOG(WARNING) << "Node  " << node_name << " is not an output node";
    std::vector<mindspore::tensor::MSTensor *> empty_ret;
    return empty_ret;
  }
  return ret->second;
}

std::vector<std::string> LiteSession::GetOutputTensorNames() const { return this->output_tensor_names_; }

mindspore::tensor::MSTensor *LiteSession::GetOutputByTensorName(const std::string &tensor_name) const {
  auto ret = output_tensor_map_.find(tensor_name);
  if (ret == output_tensor_map_.end()) {
    MS_LOG(WARNING) << "Tensor  " << tensor_name << " is not an output node";
    return nullptr;
  }
  return ret->second;
}

std::unordered_map<std::string, mindspore::tensor::MSTensor *> LiteSession::GetOutputs() const {
  return this->output_tensor_map_;
}

int LiteSession::ResizeInputs(const std::vector<mindspore::tensor::MSTensor *> &inputs,
                              const std::vector<std::vector<int>> &dims) {
  if (inputs.size() != inputs_.size()) {
    MS_LOG(ERROR) << "Inputs size " << inputs.size() << " is not equal to " << inputs_.size();
    return RET_PARAM_INVALID;
  }

  if (dims.size() != inputs.size()) {
    MS_LOG(ERROR) << "Input dims size " << dims.size() << " is not equal to the inputs size " << inputs.size();
    return RET_PARAM_INVALID;
  }

  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i] != inputs_[i]) {
      MS_LOG(ERROR) << "Input[" << i << "] tensor is not equal to the inputs have been saved!";
      return RET_PARAM_INVALID;
    }
    inputs_[i]->FreeData();
    inputs_[i]->set_shape(dims[i]);
  }
  return RET_OK;
}

void LiteSession::ResetInputsShape(const std::vector<std::vector<int>> &dims) {
  for (size_t i = 0; i < inputs_.size(); ++i) {
    inputs_[i]->FreeData();
    inputs_[i]->set_shape(dims[i]);
  }
}

int LiteSession::Resize(const std::vector<mindspore::tensor::MSTensor *> &inputs,
                        const std::vector<std::vector<int>> &dims) {
  bool expected = false;
  if (!is_running_.compare_exchange_strong(expected, true)) {
    MS_LOG(ERROR) << "Not support multi-threading";
    return RET_ERROR;
  }
  std::vector<std::vector<int>> old_dims;
  for (size_t i = 0; i < inputs_.size(); ++i) {
    old_dims.push_back(inputs_[i]->shape());
  }
  auto ret = ResizeInputs(inputs, dims);
  if (ret != RET_OK) {
    ResetInputsShape(old_dims);
    is_running_.store(false);
    return ret;
  }

  Scheduler scheduler(context_);
  ret = scheduler.ReSizeKernels(kernels_);
  if (ret != RET_OK) {
    ResetInputsShape(old_dims);
    auto resize_ret = scheduler.ReSizeKernels(kernels_);
    if (resize_ret != RET_OK) {
      MS_LOG(ERROR) << "restore kernel size fail!ret: " << resize_ret;
    }
    is_running_.store(false);
    return ret;
  }
  is_running_.store(false);
  return RET_OK;
}
}  // namespace lite

session::LiteSession *session::LiteSession::CreateSession(const lite::Context *context) {
  auto session = new lite::LiteSession();
  auto ret = session->Init(context);
  if (ret != mindspore::lite::RET_OK) {
    MS_LOG(ERROR) << "init sesssion failed";
    delete session;
    return nullptr;
  }
  return session;
}

session::LiteSession *session::LiteSession::CreateSession(const char *model_buf, size_t size,
                                                          const lite::Context *context) {
  auto *session = LiteSession::CreateSession(context);
  if (session == nullptr) {
    MS_LOG(ERROR) << "Create sesssion failed";
    return nullptr;
  }
  auto *model = lite::ImportFromBuffer(model_buf, size, true);
  if (model == nullptr) {
    MS_LOG(ERROR) << "Import model failed";
    return nullptr;
  }
  auto ret = session->CompileGraph(model);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Compile model failed";
    return nullptr;
  }
  model->buf = nullptr;
  delete (model);
  return session;
}
}  // namespace mindspore

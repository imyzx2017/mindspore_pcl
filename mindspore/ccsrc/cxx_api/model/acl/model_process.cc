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

#include "cxx_api/model/acl/model_process.h"
#include <algorithm>
#include <map>
#include "utils/utils.h"

namespace mindspore::api {
static DataType TransToApiType(aclDataType data_type) {
  static const std::map<aclDataType, api::DataType> data_type_map = {
    {ACL_FLOAT16, api::kMsFloat16}, {ACL_FLOAT, api::kMsFloat32}, {ACL_DOUBLE, api::kMsFloat64},
    {ACL_INT8, api::kMsInt8},       {ACL_INT16, api::kMsInt16},   {ACL_INT32, api::kMsInt32},
    {ACL_INT64, api::kMsInt64},     {ACL_UINT8, api::kMsUint8},   {ACL_UINT16, api::kMsUint16},
    {ACL_UINT32, api::kMsUint32},   {ACL_UINT64, api::kMsUint64}, {ACL_BOOL, api::kMsBool},
  };
  auto it = data_type_map.find(data_type);
  if (it == data_type_map.end()) {
    return api::kInvalidDataType;
  } else {
    return it->second;
  }
}

static void ConstructTensorDesc(const std::vector<AclTensorInfo> &acl_tensor_list, std::vector<Tensor> *tensor_list) {
  MS_EXCEPTION_IF_NULL(tensor_list);
  tensor_list->clear();

  for (size_t i = 0; i < acl_tensor_list.size(); ++i) {
    const auto &info = acl_tensor_list[i];
    Tensor tensor_desc;
    tensor_desc.SetName(info.name);
    tensor_desc.SetDataType(TransToApiType(info.data_type));
    tensor_desc.SetShape(info.dims);
    tensor_list->push_back(tensor_desc);
  }
}

Status ModelProcess::PreInitModelResource() {
  model_desc_ = aclmdlCreateDesc();
  aclError acl_ret = aclmdlGetDesc(model_desc_, model_id_);
  if (acl_ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Read model desc failed";
    return FAILED;
  }
  Status ret = InitInputsBuffer();
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "Create input buffer failed";
    return FAILED;
  }
  ret = InitOutputsBuffer();
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "Create output buffer failed";
    return FAILED;
  }
  return SUCCESS;
}

Status ModelProcess::LoadModelFromFile(const std::string &file_name, uint32_t *model_id) {
  MS_EXCEPTION_IF_NULL(model_id);
  aclError acl_ret = aclmdlLoadFromFile(file_name.c_str(), model_id);
  if (acl_ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Read model file failed, file name is " << file_name;
    return FAILED;
  }
  MS_LOG(INFO) << "Load model success " << file_name;
  model_id_ = *model_id;
  if (PreInitModelResource() != SUCCESS) {
    aclmdlUnload(model_id_);
    MS_LOG(ERROR) << "Pre init model resource failed, file name is " << file_name;
    return FAILED;
  }
  return SUCCESS;
}

Status ModelProcess::InitInputsBuffer() {
  aclError ret;
  size_t input_size = aclmdlGetNumInputs(model_desc_);
  MS_LOG(INFO) << "input_size = " << input_size;
  for (size_t i = 0; i < input_size; ++i) {
    auto buffer_size = aclmdlGetInputSizeByIndex(model_desc_, i);
    void *data_mem_buffer = nullptr;
    if (!is_run_on_device_) {  // need to copy input/output to/from device
      ret = aclrtMalloc(&data_mem_buffer, buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY);
      if (ret != ACL_ERROR_NONE) {
        MS_LOG(ERROR) << "Malloc device input buffer faild , input size " << buffer_size;
        return FAILED;
      }
    }

    aclmdlIODims dims;
    ret = aclmdlGetInputDims(model_desc_, i, &dims);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Get input shape failed";
      if (!is_run_on_device_) {
        aclrtFree(data_mem_buffer);
      }
      return FAILED;
    }
    aclDataType data_type = aclmdlGetInputDataType(model_desc_, i);
    std::vector<int64_t> shape(dims.dims, dims.dims + dims.dimCount);
    std::string input_name = aclmdlGetInputNameByIndex(model_desc_, i);
    if (input_name.empty()) {
      MS_LOG(WARNING) << "Get name of input " << i << " failed.";
    }
    MS_LOG(INFO) << "Name of input " << i << " is " << input_name;
    input_infos_.emplace_back(AclTensorInfo{data_mem_buffer, buffer_size, data_type, shape, input_name});
  }
  MS_LOG(INFO) << "Create model inputs success";
  return SUCCESS;
}

Status ModelProcess::CreateDataBuffer(void **data_mem_buffer, size_t buffer_size, aclmdlDataset *dataset) {
  MS_EXCEPTION_IF_NULL(data_mem_buffer);
  aclError ret;
  auto free_data_buffer = [this](void *dataMemBuffer) {
    if (!is_run_on_device_) {
      aclrtFree(dataMemBuffer);
    } else {
      aclrtFreeHost(dataMemBuffer);
    }
  };

  if (!is_run_on_device_) {
    ret = aclrtMalloc(data_mem_buffer, buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Malloc device buffer faild , buffer size " << buffer_size;
      return FAILED;
    }
  } else {
    ret = aclrtMallocHost(data_mem_buffer, buffer_size);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Malloc device buffer faild , buffer size " << buffer_size;
      return FAILED;
    }
  }

  auto data_buffer = aclCreateDataBuffer(*data_mem_buffer, buffer_size);
  if (data_buffer == nullptr) {
    MS_LOG(ERROR) << "Create Data Buffer failed";
    free_data_buffer(*data_mem_buffer);
    return FAILED;
  }
  ret = aclmdlAddDatasetBuffer(dataset, data_buffer);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "add data buffer failed";
    free_data_buffer(*data_mem_buffer);
    aclDestroyDataBuffer(data_buffer);
    return FAILED;
  }
  return SUCCESS;
}

Status ModelProcess::InitOutputsBuffer() {
  aclError ret;
  outputs_ = aclmdlCreateDataset();
  if (outputs_ == nullptr) {
    MS_LOG(ERROR) << "Create input dataset failed";
    return FAILED;
  }
  size_t output_size = aclmdlGetNumOutputs(model_desc_);
  MS_LOG(INFO) << "output_size = " << output_size;
  for (size_t i = 0; i < output_size; ++i) {
    auto buffer_size = aclmdlGetOutputSizeByIndex(model_desc_, i);

    void *data_mem_buffer = nullptr;
    if (CreateDataBuffer(&data_mem_buffer, buffer_size, outputs_) != SUCCESS) {
      MS_LOG(ERROR) << "add output data buffer failed, buffer size " << buffer_size;
      return FAILED;
    }
    aclmdlIODims dims;
    ret = aclmdlGetOutputDims(model_desc_, i, &dims);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Get input shape failed";
      if (!is_run_on_device_) {
        aclrtFree(data_mem_buffer);
      } else {
        aclrtFreeHost(data_mem_buffer);
      }
      return FAILED;
    }
    aclDataType data_type = aclmdlGetOutputDataType(model_desc_, i);
    std::vector<int64_t> shape(dims.dims, dims.dims + dims.dimCount);
    std::string output_name = aclmdlGetOutputNameByIndex(model_desc_, i);
    if (output_name.empty()) {
      MS_LOG(WARNING) << "Get name of output " << i << " failed.";
    }
    MS_LOG(INFO) << "Name of input " << i << " is " << output_name;
    output_infos_.emplace_back(AclTensorInfo{data_mem_buffer, buffer_size, data_type, shape, output_name});
  }
  MS_LOG(INFO) << "Create model output success";
  return SUCCESS;
}

void ModelProcess::DestroyInputsDataset() {
  if (inputs_ == nullptr) {
    return;
  }
  for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(inputs_); i++) {
    auto dataBuffer = aclmdlGetDatasetBuffer(inputs_, i);
    aclDestroyDataBuffer(dataBuffer);
  }
  aclmdlDestroyDataset(inputs_);
  inputs_ = nullptr;
}

void ModelProcess::DestroyInputsDataMem() {
  if (!is_run_on_device_) {
    for (const auto &item : input_infos_) {
      aclrtFree(item.device_data);
    }
  }
  input_infos_.clear();
}

void ModelProcess::DestroyInputsBuffer() {
  DestroyInputsDataMem();
  DestroyInputsDataset();
}

void ModelProcess::DestroyOutputsBuffer() {
  for (const auto &item : output_infos_) {
    if (!is_run_on_device_) {
      aclrtFree(item.device_data);
    } else {
      aclrtFreeHost(item.device_data);
    }
  }
  output_infos_.clear();

  if (outputs_ == nullptr) {
    return;
  }
  for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(outputs_); i++) {
    auto dataBuffer = aclmdlGetDatasetBuffer(outputs_, i);
    aclDestroyDataBuffer(dataBuffer);
  }
  aclmdlDestroyDataset(outputs_);
  outputs_ = nullptr;
}

Status ModelProcess::UnLoad() {
  auto ret = aclmdlUnload(model_id_);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Unload model failed";
    return FAILED;
  }
  if (model_desc_ != nullptr) {
    ret = aclmdlDestroyDesc(model_desc_);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Unload model failed";
      return FAILED;
    }
    model_desc_ = nullptr;
  }
  DestroyInputsBuffer();
  DestroyOutputsBuffer();
  MS_LOG(INFO) << "End unload model " << model_id_;
  return SUCCESS;
}

Status ModelProcess::CheckAndInitInput(const std::map<std::string, Buffer> &inputs) {
  aclError ret;
  inputs_ = aclmdlCreateDataset();
  // check inputs
  if (inputs.size() != input_infos_.size()) {
    MS_LOG(ERROR) << "inputs count not match, required count " << input_infos_.size() << ", given count "
                  << inputs.size();
    return INVALID_INPUTS;
  }
  for (size_t i = 0; i < input_infos_.size(); ++i) {
    const std::string &input_name = input_infos_[i].name;
    auto iter = inputs.find(input_name);
    if (iter == inputs.end()) {
      MS_LOG(ERROR) << "Model missing input " << input_name;
      return INVALID_INPUTS;
    }

    if (iter->second.DataSize() != input_infos_[i].buffer_size) {
      MS_LOG(ERROR) << "input " << i << " data size not match, required size " << input_infos_[i].buffer_size
                    << ", given count " << iter->second.DataSize();
      return INVALID_INPUTS;
    }
  }
  // copy inputs
  for (size_t i = 0; i < input_infos_.size(); ++i) {
    const auto &info = input_infos_[i];
    auto iter = inputs.find(info.name);
    if (iter == inputs.end()) {
      MS_LOG(ERROR) << "Model missing input " << info.name;
      return INVALID_INPUTS;
    }

    const auto &input = iter->second;
    const void *data = input.Data();

    void *input_buffer;
    if (!is_run_on_device_) {
      ret = aclrtMemcpy(info.device_data, info.buffer_size, data, input.DataSize(), ACL_MEMCPY_HOST_TO_DEVICE);
      if (ret != ACL_ERROR_NONE) {
        MS_LOG(ERROR) << "Acl memcpy input " << i << " data to device failed, buffer size " << input.DataSize();
        return FAILED;
      }
      input_buffer = info.device_data;
    } else {
      input_buffer = const_cast<void *>(data);
    }
    auto data_buffer = aclCreateDataBuffer(input_buffer, info.buffer_size);
    if (data_buffer == nullptr) {
      MS_LOG(ERROR) << "Create Data Buffer failed";
      return FAILED;
    }
    ret = aclmdlAddDatasetBuffer(inputs_, data_buffer);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "add data buffer failed";
      aclDestroyDataBuffer(data_buffer);
      return FAILED;
    }
  }
  return SUCCESS;
}

Status ModelProcess::CheckAndInitDvppInput(const void *dvpp_outputs_buffer_dev, size_t dvpp_outputs_buffer_size,
                                           size_t input_index) {
  aclError ret;
  inputs_ = aclmdlCreateDataset();
  // check inputs
  if (input_index >= input_infos_.size()) {
    MS_LOG(ERROR) << "inputs count not match, required count " << input_infos_.size() << ", given index "
                  << input_index;
    return INVALID_INPUTS;
  }
  if (dvpp_outputs_buffer_dev == nullptr) {
    MS_LOG(ERROR) << "input " << 0 << " cannot be null";
    return FAILED;
  }
  if (dvpp_outputs_buffer_size != input_infos_[input_index].buffer_size) {
    MS_LOG(ERROR) << "input " << 0 << " data size not match, required size " << input_infos_[input_index].buffer_size
                  << ", given count " << dvpp_outputs_buffer_size;
    return INVALID_INPUTS;
  }
  // copy inputs
  auto &info = input_infos_[input_index];
  auto data_buffer = aclCreateDataBuffer(const_cast<void *>(dvpp_outputs_buffer_dev), info.buffer_size);
  if (data_buffer == nullptr) {
    MS_LOG(ERROR) << "Create Data Buffer failed";
    return FAILED;
  }
  ret = aclmdlAddDatasetBuffer(inputs_, data_buffer);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "add data buffer failed";
    aclDestroyDataBuffer(data_buffer);
    return FAILED;
  }
  return SUCCESS;
}

Status ModelProcess::Predict(const std::map<std::string, Buffer> &inputs, std::map<std::string, Buffer> *outputs) {
  MS_EXCEPTION_IF_NULL(outputs);
  aclError acl_ret;
  Status ret = CheckAndInitInput(inputs);
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "check or init input failed";
    DestroyInputsDataset();
    return ret;  // forward status error
  }
  acl_ret = aclmdlExecute(model_id_, inputs_, outputs_);
  DestroyInputsDataset();
  if (acl_ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Execute Model Failed";
    return FAILED;
  }
  ret = BuildOutputs(outputs);
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "Build outputs faield";
    return FAILED;
  }
  MS_LOG(INFO) << "excute model success";
  return SUCCESS;
}

size_t ModelProcess::GetBatchSize() const {
  if (input_infos_.empty()) {
    MS_LOG(ERROR) << "Model is not loaded";
    return 0;
  }
  if (input_infos_[0].dims.empty()) {
    return 1;
  }
  return static_cast<size_t>(input_infos_[0].dims[0]);
}

Status ModelProcess::BuildOutputs(std::map<std::string, Buffer> *outputs) {
  MS_EXCEPTION_IF_NULL(outputs);
  aclError ret;
  // copy outputs
  outputs->clear();
  aclrtMemcpyKind kind = is_run_on_device_ ? ACL_MEMCPY_HOST_TO_HOST : ACL_MEMCPY_DEVICE_TO_HOST;
  for (size_t i = 0; i < output_infos_.size(); ++i) {
    const auto &info = output_infos_[i];
    // todo
    outputs->emplace(info.name, Buffer());
    auto output = outputs->rbegin()->second;
    if (!output.ResizeData(info.buffer_size)) {
      MS_LOG(ERROR) << "new output data buffer failed, data size " << info.buffer_size;
      return FAILED;
    }
    ret = aclrtMemcpy(output.MutableData(), output.DataSize(), info.device_data, info.buffer_size, kind);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Memcpy output " << i << " from " << (is_run_on_device_ ? "host" : "device")
                    << " to host failed, memory size " << info.buffer_size;
      return FAILED;
    }
  }
  return SUCCESS;
}

Status ModelProcess::GetInputsInfo(std::vector<Tensor> *tensor_list) const {
  ConstructTensorDesc(input_infos_, tensor_list);
  return SUCCESS;
}

Status ModelProcess::GetOutputsInfo(std::vector<Tensor> *tensor_list) const {
  ConstructTensorDesc(output_infos_, tensor_list);
  return SUCCESS;
}
}  // namespace mindspore::api

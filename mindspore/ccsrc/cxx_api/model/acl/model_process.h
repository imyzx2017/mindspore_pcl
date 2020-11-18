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

#ifndef MINDSPORE_CCSRC_CXXAPI_SESSION_ACL_MODEL_PROCESS_H
#define MINDSPORE_CCSRC_CXXAPI_SESSION_ACL_MODEL_PROCESS_H
#include <vector>
#include <string>
#include <map>
#include "acl/acl.h"
#include "acl/acl_mdl.h"
#include "acl/acl_rt.h"
#include "include/api/status.h"
#include "include/api/types.h"

namespace mindspore::api {
struct AclTensorInfo {
  void *device_data;
  size_t buffer_size;
  aclDataType data_type;
  std::vector<int64_t> dims;
  std::string name;
};

struct ImagesDvppOutput {
  void *buffer_device = nullptr;
  size_t buffer_size = 0;
  size_t input_index = 0;
};

class ModelProcess {
 public:
  ModelProcess()
      : model_id_(0xffffffff),
        is_run_on_device_(false),
        model_desc_(nullptr),
        inputs_(nullptr),
        outputs_(nullptr),
        input_infos_(),
        output_infos_() {}
  ~ModelProcess() {}
  Status LoadModelFromFile(const std::string &file_name, uint32_t *model_id);
  Status UnLoad();
  Status Predict(const std::map<std::string, Buffer> &inputs, std::map<std::string, Buffer> *outputs);
  Status PreInitModelResource();
  Status GetInputsInfo(std::vector<Tensor> *tensor_list) const;
  Status GetOutputsInfo(std::vector<Tensor> *tensor_list) const;

  // override this method to avoid request/reply data copy
  void SetIsDevice(bool is_device) { is_run_on_device_ = is_device; }

  size_t GetBatchSize() const;
  void set_model_id(uint32_t model_id) { model_id_ = model_id; }
  uint32_t model_id() const { return model_id_; }

 private:
  Status CreateDataBuffer(void **data_mem_buffer, size_t buffer_size, aclmdlDataset *dataset);
  Status CheckAndInitInput(const std::map<std::string, Buffer> &inputs);
  Status CheckAndInitDvppInput(const void *dvpp_outputs_buffer_dev, size_t dvpp_outputs_buffer_size,
                               size_t input_index);
  Status BuildOutputs(std::map<std::string, Buffer> *outputs);
  Status InitInputsBuffer();
  Status InitOutputsBuffer();

  void DestroyInputsDataset();
  void DestroyInputsDataMem();
  void DestroyInputsBuffer();
  void DestroyOutputsBuffer();

  uint32_t model_id_;
  // if run one device(AICPU), there is no need to alloc device memory and copy inputs to(/outputs from) device
  bool is_run_on_device_;
  aclmdlDesc *model_desc_;
  aclmdlDataset *inputs_;
  aclmdlDataset *outputs_;
  std::vector<AclTensorInfo> input_infos_;
  std::vector<AclTensorInfo> output_infos_;
};
}  // namespace mindspore::api

#endif  // MINDSPORE_CCSRC_CXXAPI_SESSION_ACL_MODEL_PROCESS_H

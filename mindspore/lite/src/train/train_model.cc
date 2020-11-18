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
#include "src/ops/primitive_c.h"
#include "include/train_model.h"
#include "src/common/log_adapter.h"
#include "include/errorcode.h"
#include "src/common/graph_util.h"
#include "src/model_common.h"

namespace mindspore::lite {

TrainModel *TrainModel::Import(const char *model_buf, size_t size) {
  if (model_buf == nullptr) {
    MS_LOG(ERROR) << "The model buf is nullptr";
    return nullptr;
  }
  flatbuffers::Verifier verify((const uint8_t *)model_buf, size);
  int schema_version = VersionVerify(&verify);
  if (schema_version == -1) {
    MS_LOG(ERROR) << "The buffer is invalid and fail to create graph.";
    return nullptr;
  }
  TrainModel *model = new (std::nothrow) TrainModel();
  if (model == nullptr) {
    MS_LOG(ERROR) << "new model fail!";
    return nullptr;
  }
  model->buf = reinterpret_cast<char *>(malloc(size));
  if (model->buf == nullptr) {
    delete model;
    MS_LOG(ERROR) << "malloc inner model buf fail!";
    return nullptr;
  }
  memcpy(model->buf, model_buf, size);
  model->buf_size_ = size;
  const void *meta_graph = GetMetaGraphByVerison(model->buf, schema_version);
  if (meta_graph == nullptr) {
    MS_LOG(ERROR) << "meta_graph is nullptr!";
    delete (model);
    return nullptr;
  }

  int status = GenerateModelByVersion(meta_graph, model, schema_version);
  if (status != RET_OK) {
    delete (model);
    MS_LOG(ERROR) << "fail to generate model";
    return nullptr;
  }
  return model;
}

void TrainModel::Free() {}

char *TrainModel::ExportBuf(char *buffer, size_t *len) const {
  if (len == nullptr) {
    MS_LOG(ERROR) << "len is nullptr";
    return nullptr;
  }
  if (buf_size_ == 0 || buf == nullptr) {
    MS_LOG(ERROR) << "Model::Export is only available for Train Session";
    return nullptr;
  }

  if (*len < buf_size_ && buffer != nullptr) {
    MS_LOG(ERROR) << "Buffer is too small, Export Failed";
    return nullptr;
  }
  if (buffer == nullptr) {
    buffer = reinterpret_cast<char *>(malloc(buf_size_));
  }
  if (buffer == nullptr) {
    MS_LOG(ERROR) << "allocated model buf fail!";
    return nullptr;
  }

  memcpy(buffer, buf, buf_size_);
  *len = buf_size_;
  return buffer;
}

TrainModel::~TrainModel() { Model::Free(); }
}  // namespace mindspore::lite

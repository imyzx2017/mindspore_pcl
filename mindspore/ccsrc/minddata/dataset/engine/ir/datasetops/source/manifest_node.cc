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

#include "minddata/dataset/engine/ir/datasetops/source/manifest_node.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/manifest_op.h"

#include "minddata/dataset/util/status.h"
namespace mindspore {
namespace dataset {

ManifestNode::ManifestNode(const std::string &dataset_file, const std::string &usage,
                           const std::shared_ptr<SamplerObj> &sampler,
                           const std::map<std::string, int32_t> &class_indexing, bool decode,
                           std::shared_ptr<DatasetCache> cache)
    : DatasetNode(std::move(cache)),
      dataset_file_(dataset_file),
      usage_(usage),
      decode_(decode),
      class_index_(class_indexing),
      sampler_(sampler) {}

Status ManifestNode::ValidateParams() {
  std::vector<char> forbidden_symbols = {':', '*', '?', '"', '<', '>', '|', '`', '&', '\'', ';'};
  for (char c : dataset_file_) {
    auto p = std::find(forbidden_symbols.begin(), forbidden_symbols.end(), c);
    if (p != forbidden_symbols.end()) {
      std::string err_msg = "ManifestNode: filename should not contain :*?\"<>|`&;\'";
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }

  Path manifest_file(dataset_file_);
  if (!manifest_file.Exists()) {
    std::string err_msg = "ManifestNode: dataset file: [" + dataset_file_ + "] is invalid or not exist";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  RETURN_IF_NOT_OK(ValidateDatasetSampler("ManifestNode", sampler_));

  RETURN_IF_NOT_OK(ValidateStringValue("ManifestNode", usage_, {"train", "eval", "inference"}));

  return Status::OK();
}

std::vector<std::shared_ptr<DatasetOp>> ManifestNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  // Do internal Schema generation.
  auto schema = std::make_unique<DataSchema>();
  RETURN_EMPTY_IF_ERROR(schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kCv, 1)));
  TensorShape scalar = TensorShape::CreateScalar();
  RETURN_EMPTY_IF_ERROR(
    schema->AddColumn(ColDescriptor("label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar)));

  std::shared_ptr<ManifestOp> manifest_op;
  manifest_op =
    std::make_shared<ManifestOp>(num_workers_, rows_per_buffer_, dataset_file_, connector_que_size_, decode_,
                                 class_index_, std::move(schema), std::move(sampler_->Build()), usage_);
  RETURN_EMPTY_IF_ERROR(AddCacheOp(&node_ops));

  node_ops.push_back(manifest_op);

  return node_ops;
}

// Get the shard id of node
Status ManifestNode::GetShardId(int32_t *shard_id) {
  *shard_id = sampler_->ShardId();

  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore

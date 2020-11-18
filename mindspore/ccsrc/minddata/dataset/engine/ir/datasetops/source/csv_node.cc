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

#include "minddata/dataset/engine/ir/datasetops/source/csv_node.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/csv_op.h"

#include "minddata/dataset/util/status.h"
namespace mindspore {
namespace dataset {

// Constructor for CSVNode
CSVNode::CSVNode(const std::vector<std::string> &csv_files, char field_delim,
                 const std::vector<std::shared_ptr<CsvBase>> &column_defaults,
                 const std::vector<std::string> &column_names, int64_t num_samples, ShuffleMode shuffle,
                 int32_t num_shards, int32_t shard_id, std::shared_ptr<DatasetCache> cache)
    : DatasetNode(std::move(cache)),
      dataset_files_(csv_files),
      field_delim_(field_delim),
      column_defaults_(column_defaults),
      column_names_(column_names),
      num_samples_(num_samples),
      shuffle_(shuffle),
      num_shards_(num_shards),
      shard_id_(shard_id) {}

Status CSVNode::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateDatasetFilesParam("CSVNode", dataset_files_));

  if (field_delim_ == '"' || field_delim_ == '\r' || field_delim_ == '\n') {
    std::string err_msg = "CSVNode: The field delimiter should not be \", \\r, \\n";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (num_samples_ < 0) {
    std::string err_msg = "CSVNode: Invalid number of samples: " + std::to_string(num_samples_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  RETURN_IF_NOT_OK(ValidateDatasetShardParams("CSVNode", num_shards_, shard_id_));

  if (find(column_defaults_.begin(), column_defaults_.end(), nullptr) != column_defaults_.end()) {
    std::string err_msg = "CSVNode: column_default should not be null.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (!column_names_.empty()) {
    RETURN_IF_NOT_OK(ValidateDatasetColumnParam("CSVNode", "column_names", column_names_));
  }

  return Status::OK();
}

// Function to build CSVNode
std::vector<std::shared_ptr<DatasetOp>> CSVNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  bool shuffle_files = (shuffle_ == ShuffleMode::kGlobal || shuffle_ == ShuffleMode::kFiles);

  // CSVOp by itself is a non-mappable dataset that does not support sampling.
  // However, if a cache operator is injected at some other place higher in the tree, that cache can
  // inherit this sampler from the leaf, providing sampling support from the caching layer.
  // That is why we save the sampler here in a leaf node that does not use sampling.
  std::shared_ptr<SamplerObj> sampler_ = SelectSampler(num_samples_, shuffle_files, num_shards_, shard_id_);

  // Sort the dataset files in a lexicographical order
  std::vector<std::string> sorted_dataset_files = dataset_files_;
  std::sort(sorted_dataset_files.begin(), sorted_dataset_files.end());

  std::vector<std::shared_ptr<CsvOp::BaseRecord>> column_default_list;
  for (auto v : column_defaults_) {
    if (v->type == CsvType::INT) {
      column_default_list.push_back(
        std::make_shared<CsvOp::Record<int>>(CsvOp::INT, std::dynamic_pointer_cast<CsvRecord<int>>(v)->value));
    } else if (v->type == CsvType::FLOAT) {
      column_default_list.push_back(
        std::make_shared<CsvOp::Record<float>>(CsvOp::FLOAT, std::dynamic_pointer_cast<CsvRecord<float>>(v)->value));
    } else if (v->type == CsvType::STRING) {
      column_default_list.push_back(std::make_shared<CsvOp::Record<std::string>>(
        CsvOp::STRING, std::dynamic_pointer_cast<CsvRecord<std::string>>(v)->value));
    }
  }

  std::shared_ptr<CsvOp> csv_op =
    std::make_shared<CsvOp>(sorted_dataset_files, field_delim_, column_default_list, column_names_, num_workers_,
                            rows_per_buffer_, num_samples_, worker_connector_size_, connector_que_size_, shuffle_files,
                            num_shards_, shard_id_, std::move(sampler_->Build()));
  RETURN_EMPTY_IF_ERROR(csv_op->Init());
  if (cache_ == nullptr && shuffle_ == ShuffleMode::kGlobal) {
    // Inject ShuffleOp
    std::shared_ptr<DatasetOp> shuffle_op = nullptr;
    int64_t num_rows = 0;

    // First, get the number of rows in the dataset
    RETURN_EMPTY_IF_ERROR(CsvOp::CountAllFileRows(sorted_dataset_files, column_names_.empty(), &num_rows));

    // Add the shuffle op after this op
    RETURN_EMPTY_IF_ERROR(AddShuffleOp(sorted_dataset_files.size(), num_shards_, num_rows, 0, connector_que_size_,
                                       rows_per_buffer_, &shuffle_op));

    node_ops.push_back(shuffle_op);
  }
  RETURN_EMPTY_IF_ERROR(AddCacheOp(&node_ops));

  node_ops.push_back(csv_op);

  return node_ops;
}

// Get the shard id of node
Status CSVNode::GetShardId(int32_t *shard_id) {
  *shard_id = shard_id_;

  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore

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

#include "minddata/dataset/engine/ir/datasetops/source/album_node.h"

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/album_op.h"

#include "minddata/dataset/util/status.h"
namespace mindspore {
namespace dataset {

// Constructor for AlbumNode
AlbumNode::AlbumNode(const std::string &dataset_dir, const std::string &data_schema,
                     const std::vector<std::string> &column_names, bool decode,
                     const std::shared_ptr<SamplerObj> &sampler, const std::shared_ptr<DatasetCache> &cache)
    : DatasetNode(std::move(cache)),
      dataset_dir_(dataset_dir),
      schema_path_(data_schema),
      column_names_(column_names),
      decode_(decode),
      sampler_(sampler) {}

Status AlbumNode::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("AlbumNode", dataset_dir_));

  RETURN_IF_NOT_OK(ValidateDatasetFilesParam("AlbumNode", {schema_path_}));

  RETURN_IF_NOT_OK(ValidateDatasetSampler("AlbumNode", sampler_));

  if (!column_names_.empty()) {
    RETURN_IF_NOT_OK(ValidateDatasetColumnParam("AlbumNode", "column_names", column_names_));
  }

  return Status::OK();
}

// Function to build AlbumNode
std::vector<std::shared_ptr<DatasetOp>> AlbumNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  auto schema = std::make_unique<DataSchema>();
  RETURN_EMPTY_IF_ERROR(schema->LoadSchemaFile(schema_path_, column_names_));

  // Argument that is not exposed to user in the API.
  std::set<std::string> extensions = {};

  RETURN_EMPTY_IF_ERROR(AddCacheOp(&node_ops));

  node_ops.push_back(std::make_shared<AlbumOp>(num_workers_, rows_per_buffer_, dataset_dir_, connector_que_size_,
                                               decode_, extensions, std::move(schema), std::move(sampler_->Build())));
  return node_ops;
}

// Get the shard id of node
Status AlbumNode::GetShardId(int32_t *shard_id) {
  *shard_id = sampler_->ShardId();

  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore

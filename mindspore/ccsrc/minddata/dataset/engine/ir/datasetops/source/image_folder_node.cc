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

#include "minddata/dataset/engine/ir/datasetops/source/image_folder_node.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/image_folder_op.h"

#include "minddata/dataset/util/status.h"
namespace mindspore {
namespace dataset {

ImageFolderNode::ImageFolderNode(std::string dataset_dir, bool decode, std::shared_ptr<SamplerObj> sampler,
                                 bool recursive, std::set<std::string> extensions,
                                 std::map<std::string, int32_t> class_indexing,
                                 std::shared_ptr<DatasetCache> cache = nullptr)
    : dataset_dir_(dataset_dir),
      decode_(decode),
      sampler_(sampler),
      recursive_(recursive),
      class_indexing_(class_indexing),
      exts_(extensions),
      DatasetNode(std::move(cache)) {}

Status ImageFolderNode::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("ImageFolderNode", dataset_dir_));

  RETURN_IF_NOT_OK(ValidateDatasetSampler("ImageFolderNode", sampler_));

  return Status::OK();
}

std::vector<std::shared_ptr<DatasetOp>> ImageFolderNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  // Do internal Schema generation.
  // This arg is exist in ImageFolderOp, but not externalized (in Python API).
  std::unique_ptr<DataSchema> schema = std::make_unique<DataSchema>();
  TensorShape scalar = TensorShape::CreateScalar();
  RETURN_EMPTY_IF_ERROR(
    schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
  RETURN_EMPTY_IF_ERROR(
    schema->AddColumn(ColDescriptor("label", DataType(DataType::DE_INT32), TensorImpl::kFlexible, 0, &scalar)));

  RETURN_EMPTY_IF_ERROR(AddCacheOp(&node_ops));

  node_ops.push_back(std::make_shared<ImageFolderOp>(num_workers_, rows_per_buffer_, dataset_dir_, connector_que_size_,
                                                     recursive_, decode_, exts_, class_indexing_, std::move(schema),
                                                     std::move(sampler_->Build())));
  return node_ops;
}

// Get the shard id of node
Status ImageFolderNode::GetShardId(int32_t *shard_id) {
  *shard_id = sampler_->ShardId();

  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore

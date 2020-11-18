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

#include "minddata/dataset/engine/ir/datasetops/source/voc_node.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/voc_op.h"

#include "minddata/dataset/util/status.h"
namespace mindspore {
namespace dataset {

// Constructor for VOCNode
VOCNode::VOCNode(const std::string &dataset_dir, const std::string &task, const std::string &usage,
                 const std::map<std::string, int32_t> &class_indexing, bool decode, std::shared_ptr<SamplerObj> sampler,
                 std::shared_ptr<DatasetCache> cache)
    : DatasetNode(std::move(cache)),
      dataset_dir_(dataset_dir),
      task_(task),
      usage_(usage),
      class_index_(class_indexing),
      decode_(decode),
      sampler_(sampler) {}

Status VOCNode::ValidateParams() {
  Path dir(dataset_dir_);

  RETURN_IF_NOT_OK(ValidateDatasetDirParam("VOCNode", dataset_dir_));

  RETURN_IF_NOT_OK(ValidateDatasetSampler("VOCNode", sampler_));

  if (task_ == "Segmentation") {
    if (!class_index_.empty()) {
      std::string err_msg = "VOCNode: class_indexing is invalid in Segmentation task.";
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
    Path imagesets_file = dir / "ImageSets" / "Segmentation" / usage_ + ".txt";
    if (!imagesets_file.Exists()) {
      std::string err_msg = "VOCNode: Invalid usage: " + usage_ + ", file does not exist";
      MS_LOG(ERROR) << "VOCNode: Invalid usage: " << usage_ << ", file \"" << imagesets_file << "\" does not exist!";
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  } else if (task_ == "Detection") {
    Path imagesets_file = dir / "ImageSets" / "Main" / usage_ + ".txt";
    if (!imagesets_file.Exists()) {
      std::string err_msg = "VOCNode: Invalid usage: " + usage_ + ", file does not exist";
      MS_LOG(ERROR) << "VOCNode: Invalid usage: " << usage_ << ", file \"" << imagesets_file << "\" does not exist!";
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  } else {
    std::string err_msg = "VOCNode: Invalid task: " + task_;
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

// Function to build VOCNode
std::vector<std::shared_ptr<DatasetOp>> VOCNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  auto schema = std::make_unique<DataSchema>();
  VOCOp::TaskType task_type_;

  if (task_ == "Segmentation") {
    task_type_ = VOCOp::TaskType::Segmentation;
    RETURN_EMPTY_IF_ERROR(schema->AddColumn(
      ColDescriptor(std::string(kColumnImage), DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
    RETURN_EMPTY_IF_ERROR(schema->AddColumn(
      ColDescriptor(std::string(kColumnTarget), DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
  } else if (task_ == "Detection") {
    task_type_ = VOCOp::TaskType::Detection;
    RETURN_EMPTY_IF_ERROR(schema->AddColumn(
      ColDescriptor(std::string(kColumnImage), DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
    RETURN_EMPTY_IF_ERROR(schema->AddColumn(
      ColDescriptor(std::string(kColumnBbox), DataType(DataType::DE_FLOAT32), TensorImpl::kFlexible, 1)));
    RETURN_EMPTY_IF_ERROR(schema->AddColumn(
      ColDescriptor(std::string(kColumnLabel), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
    RETURN_EMPTY_IF_ERROR(schema->AddColumn(
      ColDescriptor(std::string(kColumnDifficult), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
    RETURN_EMPTY_IF_ERROR(schema->AddColumn(
      ColDescriptor(std::string(kColumnTruncate), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
  }

  std::shared_ptr<VOCOp> voc_op;
  voc_op = std::make_shared<VOCOp>(task_type_, usage_, dataset_dir_, class_index_, num_workers_, rows_per_buffer_,
                                   connector_que_size_, decode_, std::move(schema), std::move(sampler_->Build()));
  RETURN_EMPTY_IF_ERROR(AddCacheOp(&node_ops));

  node_ops.push_back(voc_op);
  return node_ops;
}

// Get the shard id of node
Status VOCNode::GetShardId(int32_t *shard_id) {
  *shard_id = sampler_->ShardId();

  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore

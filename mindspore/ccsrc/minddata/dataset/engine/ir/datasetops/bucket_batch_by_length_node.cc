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

#include "minddata/dataset/engine/ir/datasetops/bucket_batch_by_length_node.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/bucket_batch_by_length_op.h"

#include "minddata/dataset/util/status.h"
namespace mindspore {
namespace dataset {

BucketBatchByLengthNode::BucketBatchByLengthNode(
  std::shared_ptr<DatasetNode> child, const std::vector<std::string> &column_names,
  const std::vector<int32_t> &bucket_boundaries, const std::vector<int32_t> &bucket_batch_sizes,
  std::function<TensorRow(TensorRow)> element_length_function,
  const std::map<std::string, std::pair<TensorShape, std::shared_ptr<Tensor>>> &pad_info, bool pad_to_bucket_boundary,
  bool drop_remainder)
    : column_names_(column_names),
      bucket_boundaries_(bucket_boundaries),
      bucket_batch_sizes_(bucket_batch_sizes),
      element_length_function_(element_length_function),
      pad_info_(pad_info),
      pad_to_bucket_boundary_(pad_to_bucket_boundary),
      drop_remainder_(drop_remainder) {
  this->children.push_back(child);
}

std::vector<std::shared_ptr<DatasetOp>> BucketBatchByLengthNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  std::shared_ptr<TensorOp> c_func;
  if (element_length_function_ != nullptr) {
    c_func = std::make_shared<CFuncOp>(element_length_function_);
  } else {
    c_func = nullptr;
  }
  node_ops.push_back(std::make_shared<BucketBatchByLengthOp>(column_names_, bucket_boundaries_, bucket_batch_sizes_,
                                                             c_func, pad_info_, pad_to_bucket_boundary_,
                                                             drop_remainder_, connector_que_size_));
  return node_ops;
}

Status BucketBatchByLengthNode::ValidateParams() {
  if (element_length_function_ == nullptr && column_names_.size() != 1) {
    std::string err_msg =
      "BucketBatchByLengthNode: when element_length_function is not specified, size of column_name must be 1 but is: " +
      std::to_string(column_names_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  // Check bucket_boundaries: must be positive and strictly increasing
  if (bucket_boundaries_.empty()) {
    std::string err_msg = "BucketBatchByLengthNode: bucket_boundaries cannot be empty.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  for (int i = 0; i < bucket_boundaries_.size(); i++) {
    if (bucket_boundaries_[i] <= 0) {
      std::string err_msg = "BucketBatchByLengthNode: Invalid non-positive bucket_boundaries, index: ";
      MS_LOG(ERROR)
        << "BucketBatchByLength: bucket_boundaries must only contain positive numbers. However, the element at index: "
        << i << " was: " << bucket_boundaries_[i];
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
    if (i > 0 && bucket_boundaries_[i - 1] >= bucket_boundaries_[i]) {
      std::string err_msg = "BucketBatchByLengthNode: Invalid bucket_boundaries not be strictly increasing.";
      MS_LOG(ERROR)
        << "BucketBatchByLength: bucket_boundaries must be strictly increasing. However, the elements at index: "
        << i - 1 << " and " << i << " were: " << bucket_boundaries_[i - 1] << " and " << bucket_boundaries_[i]
        << " respectively.";
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }

  if (!column_names_.empty()) {
    RETURN_IF_NOT_OK(ValidateDatasetColumnParam("BucketBatchByLengthNode", "column_names", column_names_));
  }

  // Check bucket_batch_sizes: must be positive
  if (bucket_batch_sizes_.empty()) {
    std::string err_msg = "BucketBatchByLengthNode: bucket_batch_sizes must be non-empty";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (bucket_batch_sizes_.size() != bucket_boundaries_.size() + 1) {
    std::string err_msg =
      "BucketBatchByLengthNode: bucket_batch_sizes's size must equal the size of bucket_boundaries + 1";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (std::any_of(bucket_batch_sizes_.begin(), bucket_batch_sizes_.end(), [](int i) { return i <= 0; })) {
    std::string err_msg = "BucketBatchByLengthNode: bucket_batch_sizes must only contain positive numbers.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore

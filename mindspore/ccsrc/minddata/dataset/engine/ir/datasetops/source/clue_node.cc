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

#include "minddata/dataset/engine/ir/datasetops/source/clue_node.h"

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/clue_op.h"

#include "minddata/dataset/util/status.h"
namespace mindspore {
namespace dataset {

// Constructor for CLUENode
CLUENode::CLUENode(const std::vector<std::string> clue_files, std::string task, std::string usage, int64_t num_samples,
                   ShuffleMode shuffle, int32_t num_shards, int32_t shard_id, std::shared_ptr<DatasetCache> cache)
    : DatasetNode(std::move(cache)),
      dataset_files_(clue_files),
      task_(task),
      usage_(usage),
      num_samples_(num_samples),
      shuffle_(shuffle),
      num_shards_(num_shards),
      shard_id_(shard_id) {}

Status CLUENode::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateDatasetFilesParam("CLUENode", dataset_files_));

  RETURN_IF_NOT_OK(ValidateStringValue("CLUENode", task_, {"AFQMC", "TNEWS", "IFLYTEK", "CMNLI", "WSC", "CSL"}));

  RETURN_IF_NOT_OK(ValidateStringValue("CLUENode", usage_, {"train", "test", "eval"}));

  if (num_samples_ < 0) {
    std::string err_msg = "CLUENode: Invalid number of samples: " + std::to_string(num_samples_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  RETURN_IF_NOT_OK(ValidateDatasetShardParams("CLUENode", num_shards_, shard_id_));

  return Status::OK();
}

// Function to split string based on a character delimiter
std::vector<std::string> CLUENode::split(const std::string &s, char delim) {
  std::vector<std::string> res;
  std::stringstream ss(s);
  std::string item;

  while (getline(ss, item, delim)) {
    res.push_back(item);
  }
  return res;
}

// Function to build CLUENode
std::vector<std::shared_ptr<DatasetOp>> CLUENode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  std::map<std::string, std::string> key_map;
  if (task_ == "AFQMC") {
    if (usage_ == "train") {
      key_map["sentence1"] = "sentence1";
      key_map["sentence2"] = "sentence2";
      key_map["label"] = "label";
    } else if (usage_ == "test") {
      key_map["id"] = "id";
      key_map["sentence1"] = "sentence1";
      key_map["sentence2"] = "sentence2";
    } else if (usage_ == "eval") {
      key_map["sentence1"] = "sentence1";
      key_map["sentence2"] = "sentence2";
      key_map["label"] = "label";
    }
  } else if (task_ == "CMNLI") {
    if (usage_ == "train") {
      key_map["sentence1"] = "sentence1";
      key_map["sentence2"] = "sentence2";
      key_map["label"] = "label";
    } else if (usage_ == "test") {
      key_map["id"] = "id";
      key_map["sentence1"] = "sentence1";
      key_map["sentence2"] = "sentence2";
    } else if (usage_ == "eval") {
      key_map["sentence1"] = "sentence1";
      key_map["sentence2"] = "sentence2";
      key_map["label"] = "label";
    }
  } else if (task_ == "CSL") {
    if (usage_ == "train") {
      key_map["id"] = "id";
      key_map["abst"] = "abst";
      key_map["keyword"] = "keyword";
      key_map["label"] = "label";
    } else if (usage_ == "test") {
      key_map["id"] = "id";
      key_map["abst"] = "abst";
      key_map["keyword"] = "keyword";
    } else if (usage_ == "eval") {
      key_map["id"] = "id";
      key_map["abst"] = "abst";
      key_map["keyword"] = "keyword";
      key_map["label"] = "label";
    }
  } else if (task_ == "IFLYTEK") {
    if (usage_ == "train") {
      key_map["label"] = "label";
      key_map["label_des"] = "label_des";
      key_map["sentence"] = "sentence";
    } else if (usage_ == "test") {
      key_map["id"] = "id";
      key_map["sentence"] = "sentence";
    } else if (usage_ == "eval") {
      key_map["label"] = "label";
      key_map["label_des"] = "label_des";
      key_map["sentence"] = "sentence";
    }
  } else if (task_ == "TNEWS") {
    if (usage_ == "train") {
      key_map["label"] = "label";
      key_map["label_desc"] = "label_desc";
      key_map["sentence"] = "sentence";
      key_map["keywords"] = "keywords";
    } else if (usage_ == "test") {
      key_map["id"] = "id";
      key_map["sentence"] = "sentence";
      key_map["keywords"] = "keywords";
    } else if (usage_ == "eval") {
      key_map["label"] = "label";
      key_map["label_desc"] = "label_desc";
      key_map["sentence"] = "sentence";
      key_map["keywords"] = "keywords";
    }
  } else if (task_ == "WSC") {
    if (usage_ == "train") {
      key_map["span1_index"] = "target/span1_index";
      key_map["span2_index"] = "target/span2_index";
      key_map["span1_text"] = "target/span1_text";
      key_map["span2_text"] = "target/span2_text";
      key_map["idx"] = "idx";
      key_map["label"] = "label";
      key_map["text"] = "text";
    } else if (usage_ == "test") {
      key_map["span1_index"] = "target/span1_index";
      key_map["span2_index"] = "target/span2_index";
      key_map["span1_text"] = "target/span1_text";
      key_map["span2_text"] = "target/span2_text";
      key_map["idx"] = "idx";
      key_map["text"] = "text";
    } else if (usage_ == "eval") {
      key_map["span1_index"] = "target/span1_index";
      key_map["span2_index"] = "target/span2_index";
      key_map["span1_text"] = "target/span1_text";
      key_map["span2_text"] = "target/span2_text";
      key_map["idx"] = "idx";
      key_map["label"] = "label";
      key_map["text"] = "text";
    }
  }

  ColKeyMap ck_map;
  for (auto &p : key_map) {
    ck_map.insert({p.first, split(p.second, '/')});
  }

  bool shuffle_files = (shuffle_ == ShuffleMode::kGlobal || shuffle_ == ShuffleMode::kFiles);

  // ClueOp by itself is a non-mappable dataset that does not support sampling.
  // However, if a cache operator is injected at some other place higher in the tree, that cache can
  // inherit this sampler from the leaf, providing sampling support from the caching layer.
  // That is why we save the sampler here in a leaf node that does not use sampling.
  std::shared_ptr<SamplerObj> sampler_ = SelectSampler(num_samples_, shuffle_files, num_shards_, shard_id_);

  // Sort the dataset files in a lexicographical order
  std::vector<std::string> sorted_dataset_files = dataset_files_;
  std::sort(sorted_dataset_files.begin(), sorted_dataset_files.end());

  std::shared_ptr<ClueOp> clue_op = std::make_shared<ClueOp>(
    num_workers_, rows_per_buffer_, num_samples_, worker_connector_size_, ck_map, sorted_dataset_files,
    connector_que_size_, shuffle_files, num_shards_, shard_id_, std::move(sampler_->Build()));
  RETURN_EMPTY_IF_ERROR(clue_op->Init());
  if (cache_ == nullptr && shuffle_ == ShuffleMode::kGlobal) {
    // Inject ShuffleOp
    std::shared_ptr<DatasetOp> shuffle_op = nullptr;
    int64_t num_rows = 0;

    // First, get the number of rows in the dataset
    RETURN_EMPTY_IF_ERROR(ClueOp::CountAllFileRows(sorted_dataset_files, &num_rows));

    // Add the shuffle op after this op
    RETURN_EMPTY_IF_ERROR(AddShuffleOp(sorted_dataset_files.size(), num_shards_, num_rows, 0, connector_que_size_,
                                       rows_per_buffer_, &shuffle_op));
    node_ops.push_back(shuffle_op);
  }
  RETURN_EMPTY_IF_ERROR(AddCacheOp(&node_ops));

  node_ops.push_back(clue_op);

  return node_ops;
}

// Get the shard id of node
Status CLUENode::GetShardId(int32_t *shard_id) {
  *shard_id = shard_id_;

  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore

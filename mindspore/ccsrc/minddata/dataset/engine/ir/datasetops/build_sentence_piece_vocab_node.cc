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

#include "minddata/dataset/engine/ir/datasetops/build_sentence_piece_vocab_node.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/build_sentence_piece_vocab_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

BuildSentenceVocabNode::BuildSentenceVocabNode(std::shared_ptr<DatasetNode> child,
                                               std::shared_ptr<SentencePieceVocab> vocab,
                                               const std::vector<std::string> &col_names, uint32_t vocab_size,
                                               float character_coverage, SentencePieceModel model_type,
                                               const std::unordered_map<std::string, std::string> &params)
    : vocab_(vocab),
      col_names_(col_names),
      vocab_size_(vocab_size),
      character_coverage_(character_coverage),
      model_type_(model_type),
      params_(params) {
  this->children.push_back(child);
}

// Function to build BuildSentenceVocabNode
std::vector<std::shared_ptr<DatasetOp>> BuildSentenceVocabNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  std::shared_ptr<BuildSentencePieceVocabOp> build_sentence_piece_vocab_op;
  build_sentence_piece_vocab_op = std::make_shared<BuildSentencePieceVocabOp>(
    vocab_, col_names_, vocab_size_, character_coverage_, model_type_, params_, connector_que_size_);
  node_ops.push_back(build_sentence_piece_vocab_op);
  return node_ops;
}

Status BuildSentenceVocabNode::ValidateParams() {
  if (vocab_ == nullptr) {
    std::string err_msg = "BuildSentenceVocabNode: vocab is null.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (vocab_size_ <= 0) {
    std::string err_msg =
      "BuildSentenceVocabNode: vocab_size should be positive, but got: " + std::to_string(vocab_size_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (character_coverage_ < 0.98f || character_coverage_ > 1.0f) {
    std::string err_msg = "BuildSentenceVocabNode: character_coverage should to be between 0.98 and 1.0, but got " +
                          std::to_string(character_coverage_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore

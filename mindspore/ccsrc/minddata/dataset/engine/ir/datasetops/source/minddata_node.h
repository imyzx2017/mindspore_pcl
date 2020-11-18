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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_MINDDATA_NODE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_MINDDATA_NODE_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"

namespace mindspore {
namespace dataset {

class MindDataNode : public DatasetNode {
 public:
  /// \brief Constructor
  MindDataNode(const std::vector<std::string> &dataset_files, const std::vector<std::string> &columns_list,
               const std::shared_ptr<SamplerObj> &sampler, nlohmann::json padded_sample, int64_t num_padded);

  /// \brief Constructor
  MindDataNode(const std::string &dataset_file, const std::vector<std::string> &columns_list,
               const std::shared_ptr<SamplerObj> &sampler, nlohmann::json padded_sample, int64_t num_padded);

  /// \brief Destructor
  ~MindDataNode() = default;

  /// \brief a base class override function to create the required runtime dataset op objects for this class
  /// \return The list of shared pointers to the newly created DatasetOps
  std::vector<std::shared_ptr<DatasetOp>> Build() override;

  /// \brief Parameters validation
  /// \return Status Status::OK() if all the parameters are valid
  Status ValidateParams() override;

  /// \brief Get the shard id of node
  /// \return Status Status::OK() if get shard id successfully
  Status GetShardId(int32_t *shard_id) override;

  /// \brief Build sampler chain for minddata dataset
  /// \return Status Status::OK() if input sampler is valid
  Status BuildMindDatasetSamplerChain(const std::shared_ptr<SamplerObj> &sampler,
                                      std::vector<std::shared_ptr<mindrecord::ShardOperator>> *operators_,
                                      int64_t num_padded);

  /// \brief Set sample_bytes when padded_sample has py::byte value
  /// \note Pybind will use this function to set sample_bytes into MindDataNode
  void SetSampleBytes(std::map<std::string, std::string> *sample_bytes);

 private:
  std::string dataset_file_;                // search_for_pattern_ will be true in this mode
  std::vector<std::string> dataset_files_;  // search_for_pattern_ will be false in this mode
  bool search_for_pattern_;
  std::vector<std::string> columns_list_;
  std::shared_ptr<SamplerObj> sampler_;
  nlohmann::json padded_sample_;
  std::map<std::string, std::string> sample_bytes_;  // enable in python
  int64_t num_padded_;
};

}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_SOURCE_MINDDATA_NODE_H_

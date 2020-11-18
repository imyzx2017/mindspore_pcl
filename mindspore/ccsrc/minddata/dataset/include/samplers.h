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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_SAMPLERS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_SAMPLERS_H_

#include <memory>
#include <string>
#include <vector>

#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/datasetops/source/mindrecord_op.h"
#endif

namespace mindspore {
namespace dataset {

// Internal Sampler class forward declaration
class SamplerRT;

class SamplerObj : public std::enable_shared_from_this<SamplerObj> {
 public:
  /// \brief Constructor
  SamplerObj();

  /// \brief Destructor
  ~SamplerObj() = default;

  /// \brief Pure virtual function for derived class to implement parameters validation
  /// \return bool true if all the parameters are valid
  virtual bool ValidateParams() = 0;

  /// \brief Pure virtual function to convert a SamplerObj class into a runtime sampler object
  /// \return Shared pointers to the newly created Sampler
  virtual std::shared_ptr<SamplerRT> Build() = 0;

  /// \brief Function for derived class to get the shard id of sampler
  /// \return The shard id of the derived sampler
  virtual int64_t ShardId() { return 0; }

#ifndef ENABLE_ANDROID
  /// \brief Virtual function to convert a SamplerObj class into a runtime mindrecord sampler object,
  ///     only override by SubsetRandomSampler, PkSampler, RandomSampler, SequentialSampler, DistributedSampler
  /// \return Shared pointers to the newly created Sampler
  virtual std::shared_ptr<mindrecord::ShardOperator> BuildForMindDataset() { return nullptr; }
#endif
};

class DistributedSamplerObj;
class PKSamplerObj;
class RandomSamplerObj;
class SequentialSamplerObj;
class SubsetRandomSamplerObj;
class WeightedRandomSamplerObj;

/// Function to create a Distributed Sampler.
/// \notes A Sampler that access a shard of the dataset.
/// \param[in] num_shards - Number of shards to divide the dataset into.
/// \param[in] shard_id - Shard ID of the current shard within num_shards.
/// \param[in] shuffle - If true, the indices are shuffled.
/// \param[in] num_samples - The number of samples to draw (default to all elements).
/// \param[in] seed - The seed in use when shuffle is true.
/// \param[in] offset - The starting position where access to elements in the dataset begins.
/// \param[in] even_dist - If true, each shard would return the same number of rows (default to true).
///     If false the total rows returned by all the shards would not have overlap.
/// \return Shared pointer to the current Sampler.
std::shared_ptr<DistributedSamplerObj> DistributedSampler(int64_t num_shards, int64_t shard_id, bool shuffle = true,
                                                          int64_t num_samples = 0, uint32_t seed = 1,
                                                          int64_t offset = -1, bool even_dist = true);

/// Function to create a PK Sampler.
/// \notes Samples K elements for each P class in the dataset.
///        This will sample all classes.
/// \param[in] num_val - Number of elements to sample for each class.
/// \param[in] shuffle - If true, the class IDs are shuffled.
/// \param[in] num_samples - The number of samples to draw (default to all elements).
/// \return Shared pointer to the current Sampler.
std::shared_ptr<PKSamplerObj> PKSampler(int64_t num_val, bool shuffle = false, int64_t num_samples = 0);

/// Function to create a Random Sampler.
/// \notes Samples the elements randomly.
/// \param[in] replacement - If true, put the sample ID back for the next draw.
/// \param[in] num_samples - The number of samples to draw (default to all elements).
/// \return Shared pointer to the current Sampler.
std::shared_ptr<RandomSamplerObj> RandomSampler(bool replacement = false, int64_t num_samples = 0);

/// Function to create a Sequential Sampler.
/// \notes Samples the dataset elements sequentially, same as not having a sampler.
/// \param[in] start_index - Index to start sampling at (dafault to start at first id).
/// \param[in] num_samples - The number of samples to draw (default to all elements).
/// \return Shared pointer to the current Sampler.
std::shared_ptr<SequentialSamplerObj> SequentialSampler(int64_t start_index = 0, int64_t num_samples = 0);

/// Function to create a Subset Random Sampler.
/// \notes Samples the elements randomly from a sequence of indices.
/// \param[in] indices - A vector sequence of indices.
/// \param[in] num_samples - The number of samples to draw (default to all elements).
/// \return Shared pointer to the current Sampler.
std::shared_ptr<SubsetRandomSamplerObj> SubsetRandomSampler(std::vector<int64_t> indices, int64_t num_samples = 0);

/// Function to create a Weighted Random Sampler.
/// \notes Samples the elements from [0, len(weights) - 1] randomly with the given
///        weights (probabilities).
/// \param[in] weights - A vector sequence of weights, not necessarily summing up to 1.
/// \param[in] num_samples - The number of samples to draw (default to all elements).
/// \param[in] replacement - If true, put the sample ID back for the next draw.
/// \return Shared pointer to the current Sampler.
std::shared_ptr<WeightedRandomSamplerObj> WeightedRandomSampler(std::vector<double> weights, int64_t num_samples = 0,
                                                                bool replacement = true);

/* ####################################### Derived Sampler classes ################################# */
class DistributedSamplerObj : public SamplerObj {
 public:
  DistributedSamplerObj(int64_t num_shards, int64_t shard_id, bool shuffle, int64_t num_samples, uint32_t seed,
                        int64_t offset, bool even_dist);

  ~DistributedSamplerObj() = default;

  std::shared_ptr<SamplerRT> Build() override;

#ifndef ENABLE_ANDROID
  std::shared_ptr<mindrecord::ShardOperator> BuildForMindDataset() override;
#endif

  bool ValidateParams() override;

  /// \brief Function to get the shard id of sampler
  /// \return The shard id of sampler
  int64_t ShardId() override { return shard_id_; }

 private:
  int64_t num_shards_;
  int64_t shard_id_;
  bool shuffle_;
  int64_t num_samples_;
  uint32_t seed_;
  int64_t offset_;
  bool even_dist_;
};

class PKSamplerObj : public SamplerObj {
 public:
  PKSamplerObj(int64_t num_val, bool shuffle, int64_t num_samples);

  ~PKSamplerObj() = default;

  std::shared_ptr<SamplerRT> Build() override;

#ifndef ENABLE_ANDROID
  std::shared_ptr<mindrecord::ShardOperator> BuildForMindDataset() override;
#endif

  bool ValidateParams() override;

 private:
  int64_t num_val_;
  bool shuffle_;
  int64_t num_samples_;
};

class RandomSamplerObj : public SamplerObj {
 public:
  RandomSamplerObj(bool replacement, int64_t num_samples);

  ~RandomSamplerObj() = default;

  std::shared_ptr<SamplerRT> Build() override;

#ifndef ENABLE_ANDROID
  std::shared_ptr<mindrecord::ShardOperator> BuildForMindDataset() override;
#endif

  bool ValidateParams() override;

 private:
  bool replacement_;
  int64_t num_samples_;
};

class SequentialSamplerObj : public SamplerObj {
 public:
  SequentialSamplerObj(int64_t start_index, int64_t num_samples);

  ~SequentialSamplerObj() = default;

  std::shared_ptr<SamplerRT> Build() override;

#ifndef ENABLE_ANDROID
  std::shared_ptr<mindrecord::ShardOperator> BuildForMindDataset() override;
#endif

  bool ValidateParams() override;

 private:
  int64_t start_index_;
  int64_t num_samples_;
};

class SubsetRandomSamplerObj : public SamplerObj {
 public:
  SubsetRandomSamplerObj(std::vector<int64_t> indices, int64_t num_samples);

  ~SubsetRandomSamplerObj() = default;

  std::shared_ptr<SamplerRT> Build() override;

#ifndef ENABLE_ANDROID
  std::shared_ptr<mindrecord::ShardOperator> BuildForMindDataset() override;
#endif

  bool ValidateParams() override;

 private:
  const std::vector<int64_t> indices_;
  int64_t num_samples_;
};

class WeightedRandomSamplerObj : public SamplerObj {
 public:
  explicit WeightedRandomSamplerObj(std::vector<double> weights, int64_t num_samples = 0, bool replacement = true);

  ~WeightedRandomSamplerObj() = default;

  std::shared_ptr<SamplerRT> Build() override;

  bool ValidateParams() override;

 private:
  const std::vector<double> weights_;
  int64_t num_samples_;
  bool replacement_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_SAMPLERS_H_

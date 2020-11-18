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
#include "minddata/dataset/engine/datasetops/source/sampler/subset_random_sampler.h"

#include <algorithm>
#include <memory>
#include <random>
#include <string>

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/util/random.h"

namespace mindspore {
namespace dataset {
// Constructor.
SubsetRandomSamplerRT::SubsetRandomSamplerRT(int64_t num_samples, const std::vector<int64_t> &indices,
                                             int64_t samples_per_buffer)
    : SamplerRT(num_samples, samples_per_buffer), indices_(indices), sample_id_(0), buffer_id_(0) {}

// Initialized this Sampler.
Status SubsetRandomSamplerRT::InitSampler() {
  CHECK_FAIL_RETURN_UNEXPECTED(
    num_rows_ > 0, "Invalid parameter, num_rows must be greater than 0, but got " + std::to_string(num_rows_) + ".\n");

  // Special value of 0 for num_samples means that the user wants to sample the entire set of data.
  // In this case, the id's are provided by the user.  Cap the num_samples on the number of id's given.
  if (num_samples_ == 0 || num_samples_ > static_cast<int64_t>(indices_.size())) {
    num_samples_ = static_cast<int64_t>(indices_.size());
  }
  // Initialize random generator with seed from config manager
  rand_gen_.seed(GetSeed());

  if (samples_per_buffer_ > num_samples_) {
    samples_per_buffer_ = num_samples_;
  }

  // num_samples_ could be smaller than the total number of input id's.
  // We will shuffle the full set of id's, but only select the first num_samples_ of them later.
  std::shuffle(indices_.begin(), indices_.end(), rand_gen_);

  return Status::OK();
}

// Reset the internal variable to the initial state.
Status SubsetRandomSamplerRT::ResetSampler() {
  // Reset the internal counters.
  sample_id_ = 0;
  buffer_id_ = 0;

  // Randomized the indices again.
  rand_gen_.seed(GetSeed());
  std::shuffle(indices_.begin(), indices_.end(), rand_gen_);

  if (HasChildSampler()) {
    RETURN_IF_NOT_OK(child_[0]->ResetSampler());
  }

  return Status::OK();
}

// Get the sample ids.
Status SubsetRandomSamplerRT::GetNextSample(std::unique_ptr<DataBuffer> *out_buffer) {
  // All samples have been drawn
  if (sample_id_ == num_samples_) {
    (*out_buffer) = std::make_unique<DataBuffer>(buffer_id_++, DataBuffer::kDeBFlagEOE);
  } else {
    if (HasChildSampler()) {
      RETURN_IF_NOT_OK(child_[0]->GetNextSample(&child_ids_));
    }

    (*out_buffer) = std::make_unique<DataBuffer>(buffer_id_++, DataBuffer::kDeBFlagNone);
    std::shared_ptr<Tensor> outputIds;

    int64_t last_id = sample_id_ + samples_per_buffer_;
    // Handling the return all samples at once, and when last draw is not a full batch.
    if (last_id > num_samples_) {
      last_id = num_samples_;
    }

    // Allocate tensor
    RETURN_IF_NOT_OK(CreateSamplerTensor(&outputIds, last_id - sample_id_));

    // Initialize tensor
    auto id_ptr = outputIds->begin<int64_t>();
    while (sample_id_ < last_id) {
      if (indices_[sample_id_] >= num_rows_) {
        std::string err_msg = "Generated indice is out of bound, expect range [0, num_data-1], got indice: " +
                              std::to_string(indices_[sample_id_]) + ", num_data: " + std::to_string(num_rows_ - 1);
        RETURN_STATUS_UNEXPECTED(err_msg);
      }

      int64_t sampled_id = ((indices_[sample_id_] % num_rows_) + num_rows_) % num_rows_;
      if (HasChildSampler()) {
        RETURN_IF_NOT_OK(GetAssociatedChildId(&sampled_id, sampled_id));
      }

      *id_ptr = sampled_id;
      id_ptr++;
      sample_id_++;
    }

    // Create a TensorTable from that single tensor and push into DataBuffer
    (*out_buffer)->set_tensor_table(std::make_unique<TensorQTable>(1, TensorRow(1, outputIds)));
  }

  return Status::OK();
}

void SubsetRandomSamplerRT::Print(std::ostream &out, bool show_all) const {
  out << "\nSampler: SubsetRandomSampler";
  if (show_all) {
    // Call the super class for displaying any common detailed info
    SamplerRT::Print(out, show_all);
    // Then add our own info if any
  }
}
}  // namespace dataset
}  // namespace mindspore

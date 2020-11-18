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

#include "backend/kernel_compiler/cpu/map_cache_idx_cpu_kernel.h"
#include <string>
#include <memory>
#include <vector>
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
template <typename T>
struct HashmapEntry {
  T key;
  T value;
  T step;
  T tag;

  bool IsEmpty() {
    if (this->tag == NULLTAG)
      return true;
    else
      return false;
  }

  bool IsUsing(const T &train_step) {
    if (this->step >= (train_step - 1))
      return true;
    else
      return false;
  }

  bool IsKey(const T &emb_idx) {
    if (this->key == emb_idx)
      return true;
    else
      return false;
  }

  void SetEmpty() { this->tag = NULLTAG; }
};

template <typename T>
T HashFunc(const T &key, const size_t &m) {
  return (T)(((0.6180339 * key) - floor(0.6180339 * key)) * m);
}

template <typename T>
int Compress(HashmapEntry<T> *entry_p, const size_t &length, T entry) {
  T i = (entry + 1) % length, off = 1;
  int compress_count = 0;
  for (; !entry_p[i].IsEmpty(); i = (i + 1) % length, off++) {
    if (entry_p[i].tag > off) {
      entry_p[entry].key = entry_p[i].key;
      entry_p[entry].value = entry_p[i].value;
      entry_p[entry].step = entry_p[i].step;
      entry_p[entry].tag = entry_p[i].tag - off;
      entry_p[i].SetEmpty();
      off = 0;
      entry = i;
    }
    compress_count++;
  }
  return compress_count;
}

void MapCacheIdxCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  node_ = kernel_node;
  auto hashmap_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);

  if (hashmap_shape.size() != 2) {
    MS_LOG(EXCEPTION) << "Dimension of HashMap must be 2, (n, 4)";
  }

  hashmap_length_ = hashmap_shape[0];
  dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0);
}

bool MapCacheIdxCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                  const std::vector<kernel::AddressPtr> & /*workspace*/,
                                  const std::vector<kernel::AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeInt32) {
    LaunchKernel<int>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt64) {
    LaunchKernel<int64_t>(inputs, outputs);
  } else {
    MS_LOG(ERROR) << "Only support int32, int64";
    return false;
  }
  return true;
}

template <typename T>
void MapCacheIdxCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  auto emb_idx_shape = AnfAlgo::GetPrevNodeOutputInferShape(node_, 1);
  batch_size_ = 1;
  for (size_t i = 0; i < emb_idx_shape.size(); ++i) {
    batch_size_ *= emb_idx_shape[i];
  }
  HashmapEntry<T> *hashmap = reinterpret_cast<HashmapEntry<T> *>(inputs[0]->addr);
  auto input_indices = reinterpret_cast<T *>(inputs[1]->addr);
  T *step_ = reinterpret_cast<T *>(inputs[2]->addr);
  T emb_max_num = *reinterpret_cast<T *>(inputs[3]->addr);
  T offset = *reinterpret_cast<T *>(inputs[4]->addr);
  auto output_cache_idx = reinterpret_cast<T *>(outputs[0]->addr);
  auto output_old_emb_idx = reinterpret_cast<T *>(outputs[1]->addr);
  auto output_miss_emb_idx = reinterpret_cast<T *>(outputs[2]->addr);
  auto output_swap_cache_idx = reinterpret_cast<T *>(outputs[3]->addr);

  std::vector<T> miss_idx;
  size_t miss_count = 0;
  float total_count = 0;
  int count_size = 0;
  float hit_count = 0;
  // search_cache_idx
  for (size_t i = 0; i < batch_size_; ++i) {
    T key = input_indices[i] - offset;
    if (key >= emb_max_num || key < 0) {
      output_cache_idx[i] = -1;
      continue;
    }

    T tmp_entry = HashFunc(key, hashmap_length_);

    size_t count = 1;
    count_size += 1;
    while ((!hashmap[tmp_entry].IsEmpty() && !hashmap[tmp_entry].IsKey(key))) {
      tmp_entry = (tmp_entry + 1) % hashmap_length_;
      if (count > hashmap_length_) {
        MS_LOG(ERROR) << "Hashmap is full, search cache idx failed!";
        break;
      }
      count += 1;
    }

    total_count += count;
    if (hashmap[tmp_entry].IsEmpty()) {
      miss_idx.emplace_back(i);
      output_miss_emb_idx[miss_count] = key;
      output_cache_idx[i] = -1;
      miss_count++;
    } else {
      hit_count += 1;
      output_cache_idx[i] = hashmap[tmp_entry].value;
      hashmap[tmp_entry].step = step_[0];
    }
  }
  MS_LOG(INFO) << "Miss count: " << miss_count;
  MS_LOG(INFO) << "Avg search count: " << total_count / count_size;
  MS_LOG(INFO) << "Cache hit rate: " << hit_count / count_size;

  float total_insert_count = 0;
  float total_delete_count = 0;

  // swap hash map
  for (size_t i = 0; i < miss_count; ++i) {
    T emb_idx = output_miss_emb_idx[i];
    T entry = HashFunc(emb_idx, hashmap_length_);
    size_t tag_count = 1;
    while (!hashmap[entry].IsEmpty()) {
      entry = (entry + 1) % hashmap_length_;
      if (tag_count > hashmap_length_) {
        MS_LOG(ERROR) << "Hashmap is full, insert new key failed!";
        break;
      }
      tag_count++;
    }

    hashmap[entry].key = emb_idx;
    hashmap[entry].step = step_[0];
    hashmap[entry].tag = tag_count;

    T tmp_entry = (entry + 1) % hashmap_length_;
    size_t delete_count = 1;
    while (hashmap[tmp_entry].IsEmpty() || hashmap[tmp_entry].IsUsing(step_[0])) {
      tmp_entry = (tmp_entry + 1) % hashmap_length_;
      if (delete_count > hashmap_length_) {
        MS_LOG(ERROR) << "Hashmap is full, delete old key failed!";
        break;
      }
      delete_count++;
    }

    output_swap_cache_idx[i] = hashmap[tmp_entry].value;
    output_old_emb_idx[i] = hashmap[tmp_entry].key;
    hashmap[entry].value = output_swap_cache_idx[i];
    hashmap[tmp_entry].SetEmpty();
    int compress_count = Compress(hashmap, hashmap_length_, tmp_entry);
    total_delete_count += (compress_count + delete_count);
    total_insert_count += tag_count;
  }

  MS_LOG(INFO) << "Insert count: " << total_insert_count / miss_count;
  MS_LOG(INFO) << "Delete count: " << total_delete_count / miss_count;

  // update step
  step_[0] += 1;

  // update cache idx
  for (size_t i = 0; i < miss_count; ++i) {
    int idx = miss_idx[i];
    output_cache_idx[idx] = output_swap_cache_idx[i];
  }

  std::vector<size_t> out_shape;
  out_shape.emplace_back(miss_count);
  std::vector<TypeId> dtypes;
  for (size_t i = 0; i < AnfAlgo::GetOutputTensorNum(node_); i++) {
    dtypes.push_back(AnfAlgo::GetOutputInferDataType(node_, i));
  }
  AnfAlgo::SetOutputInferTypeAndShape(dtypes, {AnfAlgo::GetOutputInferShape(node_, 0), out_shape, out_shape, out_shape},
                                      node_.get());
}
}  // namespace kernel
}  // namespace mindspore

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

#ifndef MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_TASK_H_
#define MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_TASK_H_

#include <algorithm>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>
#include "minddata/mindrecord/include/common/shard_utils.h"

namespace mindspore {
namespace mindrecord {
class ShardTask {
 public:
  ShardTask();

  ShardTask(const ShardTask &task);  // copy construction

  ShardTask &operator=(const ShardTask &task);  // assignment operator

  ~ShardTask() = default;

  void MakePerm();

  void InsertTask(TaskType task_type, int shard_id, int group_id, const std::vector<uint64_t> &offset,
                  const json &label);

  void InsertTask(std::tuple<TaskType, std::tuple<int, int>, std::vector<uint64_t>, json> task);

  void PopBack();

  uint32_t Size() const;

  uint32_t SizeOfRows() const;

  std::tuple<TaskType, std::tuple<int, int>, std::vector<uint64_t>, json> &GetTaskByID(size_t id);

  std::tuple<TaskType, std::tuple<int, int>, std::vector<uint64_t>, json> &GetRandomTask();

  static ShardTask Combine(std::vector<ShardTask> &category_tasks, bool replacement, int64_t num_elements,
                           int64_t num_samples);

  uint32_t categories;

  std::vector<int> permutation_;

  std::vector<std::tuple<TaskType, std::tuple<int, int>, std::vector<uint64_t>, json>> task_list_;
};
}  // namespace mindrecord
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_TASK_H_

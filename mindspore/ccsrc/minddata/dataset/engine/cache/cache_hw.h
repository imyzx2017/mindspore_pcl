/**
 * Copyright 2020 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_HW_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_HW_H_

#ifdef NUMA_ENABLED
#include <numa.h>
#endif
#include <sched.h>
#include <stdlib.h>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "minddata/dataset/engine/cache/cache_common.h"
#include "minddata/dataset/util/memory_pool.h"
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/util/task.h"

namespace mindspore {
namespace dataset {
class CacheServerHW {
 public:
  CacheServerHW();
  ~CacheServerHW() = default;

  /// \brief Get Numa node info without using numa library
  /// \return Status object
  Status GetNumaNodeInfo();

  /// \brief Set thread affinity
  Status SetAffinity(const Task &tk, numa_id_t numa_node);

  /// \brief Get total number of cpu(s)
  int32_t GetCpuCount() const { return num_cpus_; }

  /// \brief Get total number of numa nodes
  int32_t GetNumaNodeCount() const { return numa_cpuset_.empty() ? 1 : numa_cpuset_.size(); }

  /// \brief Get a list of cpu for a given numa node.
  std::vector<cpu_id_t> GetCpuList(numa_id_t numa_id);

  static bool numa_enabled();

  /// \brief Return the numa the current thread is running on.
  numa_id_t GetMyNode() const;

  /// \brief Interleave a given memory block. Used by shared memory only.
  static void InterleaveMemory(void *ptr, size_t sz);

  /// \brief Set default memory policy.
  static Status SetDefaultMemoryPolicy(CachePoolPolicy);

  /// \brief This returns the size (in bytes) of the physical RAM on the machine.
  /// \return the size (in bytes) of the physical RAM on the machine.
  static int64_t GetTotalSystemMemory();

 private:
  constexpr static char kSysNodePath[] = "/sys/devices/system/node";
  int32_t num_cpus_;
  std::map<numa_id_t, cpu_set_t> numa_cpuset_;
  std::map<numa_id_t, int32_t> numa_cpu_cnt_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_HW_H_

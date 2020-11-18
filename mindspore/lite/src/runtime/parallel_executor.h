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

#ifndef MINDSPORE_LITE_PARALLEL_EXECUTOR_H_
#define MINDSPORE_LITE_PARALLEL_EXECUTOR_H_

#include <vector>
#include <unordered_map>
#include "src/runtime/allocator.h"
#include "src/lite_kernel.h"
#include "include/lite_session.h"
#include "src/executor.h"

namespace mindspore::lite {
class ParallelExecutor : public Executor {
 public:
  ParallelExecutor() = default;
  virtual ~ParallelExecutor();

  int Prepare(const std::vector<kernel::LiteKernel *> &kernels) override;

  int Run(std::vector<Tensor *> &in_tensors, std::vector<Tensor *> &out_tensors,
          std::vector<kernel::LiteKernel *> &kernels, Allocator *allocator = nullptr,
          const KernelCallBack &before = nullptr, const KernelCallBack &after = nullptr) override;
  inline kernel::LiteKernel *GetReadyKernel(const int index) const { return readyKernels.at(index); }
  inline void SetResult(const int index, const int result) { results.at(index) = result; }

 private:
  std::unordered_map<kernel::LiteKernel *, size_t> refCount;
  std::vector<kernel::LiteKernel *> readyKernels;
  std::vector<int> results;
  struct ThreadPool *thread_pool_ = NULL;
};

}  // namespace mindspore::lite
#endif

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

#include <utility>
#include "src/runtime/parallel_executor.h"
#include "src/runtime/runtime_api.h"

#define MAX_THREAD_NUM 8
namespace mindspore::lite {
ParallelExecutor::~ParallelExecutor() { DestroyThreadPool(thread_pool_); }
int ParallelExecutor::Prepare(const std::vector<mindspore::kernel::LiteKernel *> &kernels) {
  thread_pool_ = CreateLiteThreadPool(MAX_THREAD_NUM, NO_BIND);
  if (thread_pool_ == nullptr) {
    MS_LOG(ERROR) << "Memory error: fail to new ThreadPool";
    return RET_ERROR;
  }
  return RET_OK;
}

static int RunKernel(void *data, int index) {
  ParallelExecutor *executor = reinterpret_cast<ParallelExecutor *>(data);
  auto kernel = executor->GetReadyKernel(index);
  auto ret = kernel->Run();
  executor->SetResult(index, ret);
  if (0 != ret) {
    MS_LOG(ERROR) << "run kernel failed, name: " << kernel->name();
    return 0;
  }

  ret = kernel->FreeWorkTensor();
  if (RET_OK != ret) {
    MS_LOG(ERROR) << "FreeWorkTensor failed, name: " << kernel->name();
    return ret;
  }
  return 0;
}

int ParallelExecutor::Run(std::vector<Tensor *> &in_tensors, std::vector<Tensor *> &out_tensors,
                          std::vector<kernel::LiteKernel *> &kernels, Allocator *allocator,
                          const KernelCallBack &before, const KernelCallBack &after) {
  MS_ASSERT(nullptr != allocator);
  for (auto &inTensor : in_tensors) {
    if (inTensor == nullptr) {
      MS_LOG(ERROR) << "Graph input tensor is nullptr";
      return RET_ERROR;
    }
    if (inTensor->GetFormat() != schema::Format::Format_NHWC) {
      MS_LOG(ERROR) << "Model input tensor should be NHWC";
      return RET_ERROR;
    }
  }
  kernel::LiteKernelUtil::InitTensorRefCount(kernels);

  for (auto kernel : kernels) {
    if (kernel->in_kernels().size() == 0) {
      readyKernels.emplace_back(kernel);
      continue;
    }
    refCount[kernel] = kernel->in_kernels().size();
  }
  std::vector<kernel::LiteKernel *> newReadyKernels;
  while (readyKernels.size() > 0) {
    results.resize(readyKernels.size(), RET_OK);
    ParallelLaunch(thread_pool_, RunKernel, this, readyKernels.size());

    if (std::find_if(results.begin(), results.end(), [](const int &ret) { return (ret != 0); }) != results.end()) {
      return RET_ERROR;
    }
    newReadyKernels.clear();
    for (auto completed : readyKernels) {
      for (auto out : completed->out_kernels()) {
        auto iter = refCount.find(out);
        if (iter == refCount.end()) {
          continue;
        }
        (iter->second)--;
        if (iter->second <= 0) {
          newReadyKernels.emplace_back(iter->first);
          refCount.erase(iter);
        }
      }

      auto ret = completed->FreeWorkTensor();
      if (RET_OK != ret) {
        MS_LOG(ERROR) << "FreeWorkTensor failed, name: " << completed->name();
        return ret;
      }
    }
    readyKernels.clear();
    readyKernels = std::move(newReadyKernels);
  }

  return RET_OK;
}

}  // namespace mindspore::lite

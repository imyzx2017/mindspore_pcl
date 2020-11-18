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

#ifndef MINDSPORE_LITE_SRC_EXECUTOR_H_
#define MINDSPORE_LITE_SRC_EXECUTOR_H_

#include <vector>
#include "src/runtime/allocator.h"
#include "src/lite_kernel.h"
#include "include/lite_session.h"

namespace mindspore::lite {
class Executor {
 public:
  Executor() = default;
  virtual ~Executor() = default;

  virtual int Prepare(const std::vector<kernel::LiteKernel *> &kernels) { return RET_OK; }

  virtual int Run(std::vector<Tensor *> &in_tensors, std::vector<Tensor *> &out_tensors,
                  std::vector<kernel::LiteKernel *> &kernels, Allocator *allocator = nullptr,
                  const KernelCallBack &before = nullptr, const KernelCallBack &after = nullptr);

 protected:
  int CheckInputs(std::vector<Tensor *> &in_tensors);

  int TransformTensorLayoutFp32(Tensor *tensor, schema::Format dst_format, Allocator *allocator = nullptr);

  int TransformTensorLayoutUint8(Tensor *tensor, schema::Format dst_format, Allocator *allocator = nullptr);

  int TransformTensorLayout(Tensor *tensor, schema::Format dst_format, Allocator *allocator = nullptr);
};
}  // namespace mindspore::lite
#endif

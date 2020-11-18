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

#ifndef MINDSPORE_CORE_UTILS_MS_EXCEPTION_H_
#define MINDSPORE_CORE_UTILS_MS_EXCEPTION_H_
#include <exception>
#include "utils/ms_utils.h"
namespace mindspore {
class MsException {
 public:
  static MsException &GetInstance() {
    static MsException instance;
    return instance;
  }

  void SetException() { exception_ptr_ = std::current_exception(); }

  void CheckException() {
    if (exception_ptr_ != nullptr) {
      auto exception_ptr = exception_ptr_;
      exception_ptr_ = nullptr;
      std::rethrow_exception(exception_ptr);
    }
  }

 private:
  MsException() = default;
  ~MsException() = default;
  DISABLE_COPY_AND_ASSIGN(MsException)

  std::exception_ptr exception_ptr_{nullptr};
};
}  // namespace mindspore

#endif  // MINDSPORE_CORE_UTILS_MS_EXCEPTION_H_

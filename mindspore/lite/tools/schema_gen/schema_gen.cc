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

#include "tools/schema_gen/schema_gen.h"
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include "include/errorcode.h"
#include "src/ops/schema_register.h"
#include "src/common/log_adapter.h"

namespace mindspore::lite {
using mindspore::lite::ops::SchemaRegisterImpl;

int SchemaGen::Init() {
  if (this->flags_ == nullptr) {
    return RET_ERROR;
  }
  MS_LOG(INFO) << "Export Path = " << flags_->export_path_;

  SchemaRegisterImpl *instance = SchemaRegisterImpl::Instance();
  if (instance == nullptr) {
    MS_LOG(ERROR) << "get instance fail!";
    return RET_ERROR;
  }

  std::string path = flags_->export_path_ + "/ops.fbs";
  if (access((path).c_str(), F_OK) == 0) {
    chmod((path).c_str(), S_IWUSR);
  }
  std::ofstream output(path, std::ofstream::binary);
  if (!output.is_open()) {
    MS_LOG(ERROR) << "Can not open file: " << path;
    return RET_ERROR;
  }
  std::string ns = "namespace mindspore.schema;\n\n";
  output.write(ns.c_str(), ns.length());
  for (auto &&func : instance->GetAllTypeDefCreateFuncs()) {
    std::string &&str = func();
    output.write(str.c_str(), str.length());
  }

  for (auto &&func : instance->GetAllOpDefCreateFuncs()) {
    std::string &&str = func();
    output.write(str.c_str(), str.length());
  }

  output.close();
  chmod(path.c_str(), S_IRUSR);
  return RET_OK;
}

int RunSchemaGen(int argc, const char **argv) {
  SchemaGenFlags flags;
  Option<std::string> err = flags.ParseFlags(argc, argv);
  if (err.IsSome()) {
    std::cerr << err.Get() << std::endl;
    std::cerr << flags.Usage() << std::endl;
    return RET_ERROR;
  }

  if (flags.help) {
    std::cerr << flags.Usage() << std::endl;
    return 0;
  }

  SchemaGen gen(&flags);
  int ret = gen.Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "schema gen fail!ret: " << ret;
  }
  return ret;
}
}  // namespace mindspore::lite

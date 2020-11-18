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
#include "core/util/option_parser.h"
#include <vector>
#include <string>
#include <cstring>
#include <iostream>
#include <iomanip>
#include "include/infer_log.h"

namespace mindspore {
namespace serving {
bool StartWith(const std::string &str, const std::string &expected) {
  return expected.empty() ||
         (str.size() >= expected.size() && memcmp(str.data(), expected.data(), expected.size()) == 0);
}

bool RemovePrefix(std::string *const str, const std::string &prefix) {
  if (!StartWith(*str, prefix)) return false;
  str->replace(str->begin(), str->begin() + prefix.size(), "");
  return true;
}

bool Option::ParseInt32(std::string *const arg) {
  if (RemovePrefix(arg, "--") && RemovePrefix(arg, name_) && RemovePrefix(arg, "=")) {
    int32_t parsed_value;
    try {
      parsed_value = std::stoi(arg->data());
    } catch (std::invalid_argument) {
      std::cout << "Parse " << name_ << " Error for option " << *arg << std::endl;
      return false;
    }
    *int32_default_ = parsed_value;
    return true;
  }
  return false;
}

bool Option::ParseBool(std::string *const arg) {
  if (RemovePrefix(arg, "--") && RemovePrefix(arg, name_) && RemovePrefix(arg, "=")) {
    if (*arg == "true") {
      *bool_default_ = true;
    } else if (*arg == "false") {
      *bool_default_ = false;
    } else {
      std::cout << "Parse " << name_ << " Error for option " << *arg << std::endl;
      return false;
    }
    return true;
  }

  return false;
}

bool Option::ParseString(std::string *const arg) {
  if (RemovePrefix(arg, "--") && RemovePrefix(arg, name_) && RemovePrefix(arg, "=")) {
    *string_default_ = *arg;
    return true;
  }
  return false;
}

bool Option::ParseFloat(std::string *const arg) {
  if (RemovePrefix(arg, "--") && RemovePrefix(arg, name_) && RemovePrefix(arg, "=")) {
    float parsed_value;
    try {
      parsed_value = std::stof(arg->data());
    } catch (std::invalid_argument) {
      std::cout << "Parse " << name_ << " Error for option " << *arg << std::endl;
      return false;
    }
    *float_default_ = parsed_value;
    return true;
  }
  return false;
}

Option::Option(const std::string &name, int32_t *const default_point, const std::string &usage)
    : name_(name),
      type_(MS_TYPE_INT32),
      int32_default_(default_point),
      bool_default_(nullptr),
      string_default_(nullptr),
      float_default_(nullptr),
      usage_(usage) {}

Option::Option(const std::string &name, bool *const default_point, const std::string &usage)
    : name_(name),
      type_(MS_TYPE_BOOL),
      int32_default_(nullptr),
      bool_default_(default_point),
      string_default_(nullptr),
      float_default_(nullptr),
      usage_(usage) {}

Option::Option(const std::string &name, std::string *const default_point, const std::string &usage)
    : name_(name),
      type_(MS_TYPE_STRING),
      int32_default_(nullptr),
      bool_default_(nullptr),
      string_default_(default_point),
      float_default_(nullptr),
      usage_(usage) {}

Option::Option(const std::string &name, float *const default_point, const std::string &usage)
    : name_(name),
      type_(MS_TYPE_FLOAT),
      int32_default_(nullptr),
      bool_default_(nullptr),
      string_default_(nullptr),
      float_default_(default_point),
      usage_(usage) {}

bool Option::Parse(std::string *const arg) {
  bool result = false;
  switch (type_) {
    case MS_TYPE_BOOL:
      result = ParseBool(arg);
      break;
    case MS_TYPE_FLOAT:
      result = ParseFloat(arg);
      break;
    case MS_TYPE_INT32:
      result = ParseInt32(arg);
      break;
    case MS_TYPE_STRING:
      result = ParseString(arg);
      break;
    default:
      break;
  }
  return result;
}

std::shared_ptr<Options> Options::inst_ = nullptr;

Options &Options::Instance() {
  static Options instance;
  return instance;
}

Options::Options() : args_(nullptr) { CreateOptions(); }

void Options::CreateOptions() {
  args_ = std::make_shared<Arguments>();
  std::vector<Option> options = {
    Option("port", &args_->grpc_port,
           "[Optional] Port to listen on for gRPC API, default is 5500, range from 1 to 65535"),
    Option("rest_api_port", &args_->rest_api_port,
           "[Optional] Port to listen on for RESTful API, default is 5501, range from 1 to 65535"),
    Option("model_name", &args_->model_name, "[Required] model name "),
    Option("model_path", &args_->model_path, "[Required] the path of the model files"),
    Option("device_id", &args_->device_id, "[Optional] the device id, default is 0, range from 0 to 7"),
  };
  options_ = options;
}

bool Options::CheckOptions() {
  if (args_->model_name == "" || args_->model_path == "") {
    std::cout << "Serving Error: model_path and model_name should not be null" << std::endl;
    return false;
  }
  if (args_->device_type != "Ascend") {
    std::cout << "Serving Error: device_type only support Ascend right now" << std::endl;
    return false;
  }
  if (args_->device_id > 7) {
    std::cout << "Serving Error: the device_id should be in [0~7]" << std::endl;
    return false;
  }
  if (args_->grpc_port < 1 || args_->grpc_port > 65535) {
    std::cout << "Serving Error: the port should be in [1~65535]" << std::endl;
    return false;
  }
  if (args_->rest_api_port < 1 || args_->rest_api_port > 65535) {
    std::cout << "Serving Error: the rest_api_port should be in [1~65535]" << std::endl;
    return false;
  }
  if (args_->rest_api_port == args_->grpc_port) {
    std::cout << "Serving Error: the rest_api_port and grpc port should not be same" << std::endl;
    return false;
  }
  return true;
}

bool Options::ParseCommandLine(int argc, char **argv) {
  if (argc < 2 || (strcmp(argv[1], "--help") == 0)) {
    Usage();
    return false;
  }
  std::vector<std::string> unkown_options;
  for (int i = 1; i < argc; ++i) {
    bool found = false;
    for (auto &option : options_) {
      std::string arg = argv[i];
      if (option.Parse(&arg)) {
        found = true;
        break;
      }
    }

    if (found == false) {
      unkown_options.push_back(argv[i]);
    }
  }

  if (!unkown_options.empty()) {
    std::cout << "unkown options:" << std::endl;
    for (const auto &option : unkown_options) {
      std::cout << option << std::endl;
    }
  }
  bool valid = (unkown_options.empty() && CheckOptions());
  if (!valid) {
    Usage();
  }
  return valid;
}

void Options::Usage() {
  std::cout << "USAGE: mindspore-serving [options]" << std::endl;

  for (const auto &option : options_) {
    std::string type;
    switch (option.type_) {
      case Option::MS_TYPE_BOOL:
        type = "bool";
        break;
      case Option::MS_TYPE_FLOAT:
        type = "float";
        break;
      case Option::MS_TYPE_INT32:
        type = "int32";
        break;
      case Option::MS_TYPE_STRING:
        type = "string";
        break;
      default:
        break;
    }
    std::cout << "--" << std::setw(30) << std::left << option.name_ << std::setw(10) << std::left << type
              << option.usage_ << std::endl;
  }
}
}  // namespace serving
}  // namespace mindspore

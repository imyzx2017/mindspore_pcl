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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TFLITE_CONVERTER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TFLITE_CONVERTER_H_

#include <string>
#include <memory>
#include <map>
#include "tools/converter/converter.h"
#include "tools/converter/parser/tflite/tflite_model_parser.h"
#include "tools/converter/graphdef_transform.h"

namespace mindspore {
namespace lite {
class TfliteConverter : public Converter {
 public:
  TfliteConverter();

  ~TfliteConverter() override = default;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TFLITE_CONVERTER_H_

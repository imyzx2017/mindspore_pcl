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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_CONVERTER_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_CONVERTER_H

#include <memory>
#include <string>
#include "schema/inner/model_generated.h"
#include "tools/converter/graphdef_transform.h"
#include "tools/converter/model_parser.h"
#include "tools/anf_importer/anf_importer.h"
#include "tools/converter/converter_flags.h"
#include "tools/converter/anf_transform.h"
#include "tools/converter/converter_context.h"

namespace mindspore {
namespace lite {
class Converter {
 public:
  Converter();
  virtual ~Converter();
  virtual schema::MetaGraphT *Convert(const lite::converter::Flags *flags);

 protected:
  ModelParser *modelParser = nullptr;
  AnfImporter *modelImporter = nullptr;
  GraphDefTransform *transform = nullptr;
  AnfTransform *anfTransform = nullptr;
};

int RunConverter(int argc, const char **argv);
}  // namespace lite
}  // namespace mindspore

#endif

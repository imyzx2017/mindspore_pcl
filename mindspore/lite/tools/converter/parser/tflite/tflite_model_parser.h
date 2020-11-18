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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TFLITE_MODEL_PARSER_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TFLITE_MODEL_PARSER_H

#include <fcntl.h>
#include <unistd.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <unordered_map>
#include "securec/include/securec.h"
#include "tools/converter/model_parser.h"
#include "tools/converter/parser/tflite/tflite_node_parser_registry.h"
#include "tools/common/tensor_util.h"
#include "schema/inner/model_generated.h"

namespace mindspore::lite {
class TfliteModelParser : public ModelParser {
 public:
  TfliteModelParser();

  ~TfliteModelParser() override;

  schema::MetaGraphT *ParseToFb(const std::string &model_file, const std::string &weight_file,
                                const QuantType &quantTyp) override;

 protected:
  std::unique_ptr<tflite::ModelT> ReadTfliteModel(const char *model_path);

  STATUS CopyConstTensorData(const std::vector<std::unique_ptr<tflite::BufferT>> &tflite_model_buffer,
                             const tflite::TensorT *tflite_tensor, schema::TensorT *tensor);

  void SetTensorQuantParam(const std::unique_ptr<tflite::TensorT> &tflite_tensor, schema::TensorT *tensor);

  STATUS ConvertOp(const std::unique_ptr<tflite::ModelT> &tflite_model,
                   const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph, const QuantType &quant_type,
                   schema::MetaGraphT *sub_graph);

  STATUS ConvertTensor(const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                       const std::vector<std::unique_ptr<tflite::BufferT>> &tflite_model_buffer,
                       schema::MetaGraphT *sub_graph);

  STATUS GetGraphInfo(const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph, schema::MetaGraphT *sub_graph);

  STATUS ConvertGroupDepthwiseOp(schema::MetaGraphT *sub_graph);

  QuantType quantType = QuantType_QUANT_NONE;
  char *tfliteModelBuf = nullptr;
  std::unique_ptr<schema::MetaGraphT> ConstructMainGraph(const std::unique_ptr<tflite::ModelT> &tflite_model,
                                                         const QuantType &quant_type);

 private:
  TfliteTensorsInfo tensorsInfo;
  std::vector<schema::TensorT *> tensors;

  std::map<std::string, schema::CNodeT *> opMap;
  std::map<const tflite::OperatorT *, schema::CNodeT *> tfliteOpMap;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TFLITE_MODEL_PARSER_H

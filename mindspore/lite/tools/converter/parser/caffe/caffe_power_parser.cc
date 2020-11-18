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

#include "tools/converter/parser/caffe/caffe_power_parser.h"
#include <memory>
#include <vector>

static const float CAFFE_POWER_DEFAULT_POWER = 1.0;
static const float CAFFE_POWER_DEFAULT_SCALE = 1.0;
static const float CAFFE_POWER_DEFAULT_SHIFT = 0.0;

namespace mindspore {
namespace lite {
STATUS CaffePowerParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight,
                               schema::CNodeT *op, std::vector<schema::TensorT *> *weightVec) {
  MS_LOG(DEBUG) << "parse CaffePowerParser";
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  std::unique_ptr<schema::PowerT> attr = std::make_unique<schema::PowerT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }

  const caffe::PowerParameter powerParam = proto.power_param();
  if (proto.has_power_param()) {
    attr->power = powerParam.has_power() ? powerParam.power() : CAFFE_POWER_DEFAULT_POWER;
    attr->scale = powerParam.has_scale() ? powerParam.scale() : CAFFE_POWER_DEFAULT_SCALE;
    attr->shift = powerParam.has_shift() ? powerParam.shift() : CAFFE_POWER_DEFAULT_SHIFT;
  } else {
    attr->power = CAFFE_POWER_DEFAULT_POWER;
    attr->scale = CAFFE_POWER_DEFAULT_SCALE;
    attr->shift = CAFFE_POWER_DEFAULT_SHIFT;
  }

  op->name = proto.name();
  op->primitive->value.type = schema::PrimitiveType_Power;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

CaffeNodeRegistrar g_caffePowerParser("Power", new CaffePowerParser());
}  // namespace lite
}  // namespace mindspore

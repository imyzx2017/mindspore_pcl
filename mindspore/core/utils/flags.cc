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

#include "utils/flags.h"
namespace mindspore {
// flag names
const char GRAPH_FLAG_MIX_PRECISION_FP16[] = "fp16";
const char GRAPH_FLAG_MIX_PRECISION_FP32[] = "fp32";
const char GRAPH_FLAG_HAS_EFFECT[] = "has_effect";
const char GRAPH_FLAG_EFFECT_PATIAL_ORDER[] = "_effect_patial_order";
const char GRAPH_FLAG_RANDOM_EFFECT[] = "_random_effect";
const char GRAPH_FLAG_SIDE_EFFECT[] = "_side_effect";

// method names of python primitive called from c++ source code
// 1. infer method name of class 'PrimitiveWithInfer'
const char PY_PRIM_METHOD_INFER[] = "__infer__";
// 2. check method name of class 'PrimitiveWithCheck'
const char PY_PRIM_METHOD_CHECK[] = "__check__";
// 3. method name of class 'PrimitivePy' for constant propagation
const char PY_PRIM_METHOD_INFER_VALUE[] = "infer_value";

// type inference related attributes
const char ATTR_VALUE[] = "value";
const char ATTR_DTYPE[] = "dtype";
const char ATTR_SHAPE[] = "shape";
const char ATTR_MIN_SHAPE[] = "min_shape";
const char ATTR_MAX_SHAPE[] = "max_shape";
}  // namespace mindspore

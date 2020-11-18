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

#include "frontend/parallel/node_check.h"

#include <set>
#include <string>

#include "frontend/parallel/ops_info/ops_utils.h"

namespace mindspore {
namespace parallel {
const std::set<std::string> BLACK_LIST = {TUPLE_GETITEM,
                                          J,
                                          LIST_GETITEM,
                                          ARRAY_GETITEM,
                                          TUPLE_SETITEM,
                                          DEPEND,
                                          LIST_SETITEM,
                                          ARRAY_SETITEM,
                                          DICT_GETITEM,
                                          LIST_APPEND,
                                          LIST_MAP,
                                          LIST_REDUCE,
                                          TUPLE_REVERSED,
                                          TILE_SHAPE,
                                          TUPLE_DIV,
                                          TUPLE_TO_ARRAY,
                                          MAKE_DICT,
                                          MAKE_SLICE,
                                          MAKE_RECORD,
                                          STRING_EQUAL,
                                          VIRTUALLOSS,
                                          RETURN,
                                          ENV_GETITEM,
                                          IDENTITY,
                                          PARTIAL,
                                          ENVSETITEM,
                                          ENVGETITEM,
                                          ENVADD,
                                          MAKEREFKEY,
                                          MAKEREF,
                                          GETREFKEY,
                                          GETREFVALUE,
                                          GETREFORIGIN,
                                          DOT,
                                          IM2COL,
                                          COL2IM,
                                          IM2COLV1,
                                          STATESETITEM,
                                          SCALARSUMMARY,
                                          IMAGESUMMARY,
                                          TENSORSUMMARY,
                                          DEBUG,
                                          HISTOGRAMSUMMARY,
                                          COL2IMV1,
                                          RESOLVE,
                                          BROADCASTGRADIENTARGS,
                                          INVERTPERMUTATION,
                                          CONTROLDEPEND,
                                          DROPOUT_GEN_MASK,
                                          EMBED,
                                          CREATINSTANCE,
                                          REF_TO_EMBED,
                                          STOP_GRADIENT};

const std::set<std::string> BATCH_PARALLEL_BLACK_LIST = {PACK, TENSOR_SCATTER_UPDATE, MIN_MAX_UPDATE_PER_LAYER};

bool IsInBlackList(const PrimitivePtr &prim) {
  MS_EXCEPTION_IF_NULL(prim);
  return (BLACK_LIST.find(prim->name()) != BLACK_LIST.end());
}

bool IsInBatchParallelBlackList(const PrimitivePtr &prim) {
  MS_EXCEPTION_IF_NULL(prim);
  return (BATCH_PARALLEL_BLACK_LIST.find(prim->name()) != BATCH_PARALLEL_BLACK_LIST.end());
}
}  // namespace parallel
}  // namespace mindspore

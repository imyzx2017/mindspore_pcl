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

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include "tools/converter/legacy_optimizer/fusion/format_trans_transpose_fusion_pass.h"
#include "src/common/log_adapter.h"
#include "securec/include/securec.h"
#include "tools/common/graph_util.h"
#include "include/errorcode.h"
#include "schema/inner/model_generated.h"

namespace mindspore {
namespace lite {
#define kFormatTransTransposeMatchPathLen 2

STATUS FormatTransPermuteFusionPass::DefinePattern() {
  // format trans + permute
  {
    auto formatTransOp = std::make_shared<PatternOp>();
    formatTransOp->id = kFormatTransformOp;
    formatTransOp->types = {PrimitiveType_Nchw2Nhwc, PrimitiveType_Nhwc2Nchw};
    auto transposeOp = std::make_shared<PatternOp>();
    transposeOp->id = kPermuteOp;
    transposeOp->types = {PrimitiveType_Transpose};

    transposeOp->left = formatTransOp;
    std::unique_ptr<FusionPattern> formatTransTransposeFusionPattern(
      new (std::nothrow) FusionPattern(kFormatTrans2TransposeFusionPattern));
    if (formatTransTransposeFusionPattern == nullptr) {
      MS_LOG(ERROR) << "new " << kFormatTrans2TransposeFusionPattern << " failed";
      return RET_ERROR;
    }
    formatTransTransposeFusionPattern->AddPatternOp(formatTransOp);
    formatTransTransposeFusionPattern->AddPatternOp(transposeOp);
    formatTransTransposeFusionPattern->Finish();
    this->patterns.emplace_back(formatTransTransposeFusionPattern.release());
  }
  // permute + format trans
  {
    auto formatTransOp = std::make_shared<PatternOp>();
    formatTransOp->id = kFormatTransformOp;
    formatTransOp->types = {PrimitiveType_Nchw2Nhwc, PrimitiveType_Nhwc2Nchw};
    auto transposeOp = std::make_shared<PatternOp>();
    transposeOp->id = kPermuteOp;
    transposeOp->types = {PrimitiveType_Transpose};

    formatTransOp->left = transposeOp;
    std::unique_ptr<FusionPattern> transposeFormatTransFusionPattern(
      new (std::nothrow) FusionPattern(kTranspose2FormatTransFusionPattern));
    if (transposeFormatTransFusionPattern == nullptr) {
      MS_LOG(ERROR) << "new " << kTranspose2FormatTransFusionPattern << " failed";
      return RET_ERROR;
    }
    transposeFormatTransFusionPattern->AddPatternOp(formatTransOp);
    transposeFormatTransFusionPattern->AddPatternOp(transposeOp);
    transposeFormatTransFusionPattern->Finish();
    this->patterns.emplace_back(transposeFormatTransFusionPattern.release());
  }
  return RET_OK;
}

STATUS FormatTransPermuteFusionPass::Run(schema::MetaGraphT *graph) { return FusionPass::Run(graph); }

STATUS FormatTransPermuteFusionPass::DoFusion(schema::MetaGraphT *graph, const std::string &patternName,
                                              std::unordered_map<std::string, std::shared_ptr<Path>> &matchedPath) {
  MS_ASSERT(graph != nullptr);
  if (matchedPath.size() != kFormatTransTransposeMatchPathLen) {
    MS_LOG(ERROR) << "schema::Format-Transform-Transpose-Fusion should have " << kFormatTransTransposeMatchPathLen
                  << " NodeIndex in matchedPair";
    return RET_PARAM_INVALID;
  }

  std::shared_ptr<Path> formatTransPath = matchedPath[kFormatTransformOp];
  std::shared_ptr<Path> transposePath = matchedPath[kPermuteOp];
  if (formatTransPath == nullptr) {
    MS_LOG(ERROR) << "formatTransPath is failed to get";
    return RET_ERROR;
  }
  if (transposePath == nullptr) {
    MS_LOG(ERROR) << "permutePath is failed to get";
    return RET_ERROR;
  }
  auto &formatTransNode = graph->nodes.at(formatTransPath->nodeIdx);
  auto &transposeNode = graph->nodes.at(transposePath->nodeIdx);
  MS_ASSERT(formatTransNode != nullptr);
  MS_ASSERT(transposeNode != nullptr);
  auto formatTransType = formatTransNode->primitive->value.type;
  if (formatTransType != PrimitiveType_Nhwc2Nchw && formatTransType != PrimitiveType_Nchw2Nhwc) {
    MS_LOG(ERROR) << "FormatTransNode should be " << EnumNamePrimitiveType(PrimitiveType_Nhwc2Nchw) << " or "
                  << EnumNamePrimitiveType(PrimitiveType_Nchw2Nhwc) << ", but got "
                  << EnumNamePrimitiveType(formatTransType);
    return RET_ERROR;
  }
  MS_ASSERT(transposeNode->primitive != nullptr);
  auto transposePrimitive = transposeNode->primitive->value.AsTranspose();
  MS_ASSERT(transposePrimitive != nullptr);
  auto perm = transposePrimitive->perm;
  if (perm.size() != 4) {
    return RET_OK;
  }
  std::vector<int32_t> nchw2nhwcPerm = {0, 2, 3, 1};
  std::vector<int32_t> nhwc2nchwPerm = {0, 3, 1, 2};
  if ((perm == nchw2nhwcPerm && formatTransType == PrimitiveType_Nhwc2Nchw) ||
      (perm == nhwc2nchwPerm && formatTransType == PrimitiveType_Nchw2Nhwc)) {
    if (formatTransPath->nodeIdx < transposePath->nodeIdx) {
      if (graph->allTensors.at(formatTransNode->inputIndex[0])->format !=
          graph->allTensors.at(transposeNode->outputIndex[0])->format) {
        return RET_OK;
      }
    } else {
      if (graph->allTensors.at(transposeNode->inputIndex[0])->format !=
          graph->allTensors.at(formatTransNode->outputIndex[0])->format) {
        return RET_OK;
      }
    }
    auto status = IsolateOneWayNode(graph, formatTransPath->nodeIdx);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "IsolateOneWayNode failed, node: " << formatTransNode->name << ", error: " << status;
      return status;
    }

    status = IsolateOneWayNode(graph, transposePath->nodeIdx);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "IsolateOneWayNode failed, node: " << transposeNode->name << ", error: " << status;
      return status;
    }
  }

  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore

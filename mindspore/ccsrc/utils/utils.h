/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_UTILS_UTILS_H_
#define MINDSPORE_CCSRC_UTILS_UTILS_H_

#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <string>
#include <vector>
#include <set>

#include "utils/log_adapter.h"
#include "ir/dtype/type.h"

namespace mindspore {
// op name. Op which not exists in operator/ops.h, so define it's name here
constexpr auto kFour2FiveOpName = "Four2Five";
constexpr auto kFive2FourOpName = "Five2Four";
constexpr auto kConv2DOpName = "Conv2D";
constexpr auto kConvBN1OpName = "ConvBN1";
constexpr auto kBN2AddReluOpName = "BN2AddRelu";
constexpr auto kBN2ReLUOpName = "BN2Relu";
constexpr auto kBN2OpName = "BN2";
constexpr auto kFusedBN1OpName = "FusedBN1";
constexpr auto kFusedBN2OpName = "FusedBN2";
constexpr auto kFusedBN3OpName = "FusedBN3";
constexpr auto kBNGrad1OpName = "BNGrad1";
constexpr auto kBNGrad2OpName = "BNGrad2";
constexpr auto kBNGrad3OpName = "BNGrad3";
constexpr auto kFusedBatchNormEx = "FusedBatchNormEx";
constexpr auto kFusedBatchNormExWithActivation = "FusedBatchNormExWithActivation";
constexpr auto kFusedBatchNormExWithAddAndActivation = "FusedBatchNormExWithAddAndActivation";
constexpr auto kFusedBatchNormGradEx = "FusedBatchNormGradEx";
constexpr auto kFusedBatchNormGradExWithActivation = "FusedBatchNormGradExWithActivation";
constexpr auto kFusedBatchNormGradExWithAddAndActivation = "FusedBatchNormGradExWithAddAndActivation";
constexpr auto kClearZeroOpName = "ClearZero";
constexpr auto kAtomicAddrCleanOpName = "AtomicAddrClean";
constexpr auto kGetNextOpName = "GetNext";
constexpr auto kEndOfSequence = "EndOfSequence";
constexpr auto kAllReduceOpName = "AllReduce";
constexpr auto kAllGatherOpName = "AllGather";
constexpr auto kHostAllGatherOpName = "HostAllGather";
constexpr auto kBroadcastOpName = "Broadcast";
constexpr auto kReduceScatterOpName = "ReduceScatter";
constexpr auto kHostReduceScatterOpName = "HostReduceScatter";
constexpr auto kMemCpyAsyncOpName = "memcpy_async";
constexpr auto kTopKOpName = "TopK";
constexpr auto kExtractImagePatchesOpName = "ExtractImagePatches";
constexpr auto kBNTrainingReduceOpName = "BNTrainingReduce";
constexpr auto kBNTrainingUpdateOpName = "BNTrainingUpdate";
constexpr auto kBNTrainingUpdateV2OpName = "BNTrainingUpdateV2";
constexpr auto kBNTrainingUpdateV3OpName = "BNTrainingUpdateV3";
constexpr auto kSimpleMeanGradOpName = "SimpleMeanGrad";
constexpr auto kMeanGradOpName = "MeanGrad";
constexpr auto kSliceOpName = "Slice";
constexpr auto kSliceGradOpName = "SliceGrad";
constexpr auto kTileOpName = "Tile";
constexpr auto kScatterNdOpName = "ScatterNd";
constexpr auto kStridedSliceAssignOpName = "StridedSliceAssign";
constexpr auto kStridedSliceOpName = "StridedSlice";
constexpr auto kStridedSliceGradOpName = "StridedSliceGrad";
constexpr auto kSparseGatherV2 = "SparseGatherV2";
constexpr auto kUnsortedSegmentProdOpName = "UnsortedSegmentProd";
constexpr auto kUnsortedSegmentMinOpName = "UnsortedSegmentMin";
constexpr auto kFlattenGradOpName = "FlattenGrad";
constexpr auto kExpandDimsOpName = "ExpandDims";
constexpr auto kSplitOpName = "Split";
constexpr auto kSplitVOpName = "SplitV";
constexpr auto kSparseApplyAdagradOpName = "SparseApplyAdagrad";
constexpr auto kMomentumOpName = "Momentum";
constexpr auto kApplyMomentumOpName = "ApplyMomentum";
constexpr auto kCombineMomentumOpName = "CombineMomentum";
constexpr auto kCombineMomentumWeightOpName = "CombineMomentumWeight";
constexpr auto kApplyAdadeltaOpName = "ApplyAdadelta";
constexpr auto kApplyAdagradOpName = "ApplyAdagrad";
constexpr auto kApplyAdagradDAName = "ApplyAdagradDA";
constexpr auto kApplyAdamOpName = "Adam";
constexpr auto kApplyAdaMaxOpName = "ApplyAdaMax";
constexpr auto kApplyAddSignOpName = "ApplyAddSign";
constexpr auto kApplyCenteredRMSPOpName = "ApplyCenteredRMSP";
constexpr auto kApplyCenteredRMSPropOpName = "ApplyCenteredRMSProp";
constexpr auto kApplyFtrlOpName = "ApplyFtrl";
constexpr auto kApplyFtrlV2OpName = "ApplyFtrlV2";
constexpr auto kApplyGradientDescentOpName = "ApplyGradientDescent";
constexpr auto kApplyPowerSignOpName = "ApplyPowerSign";
constexpr auto kApplyProximalAdagradOpName = "ApplyProximalAdagrad ";
constexpr auto kApplyProximalGradientDescentOpName = "ApplyProximalGradientDescent";
constexpr auto kApplyRMSPropOpName = "ApplyRMSProp";
constexpr auto kTransDataOpName = "TransData";
constexpr auto kBNTrainingUpdateGradOpName = "BNTrainingUpdateGrad";
constexpr auto kBNTrainingReduceGradOpName = "BNTrainingReduceGrad";
constexpr auto kSquareSumV1OpName = "SquareSumV1";
constexpr auto kSquareSumV2OpName = "SquareSumV2";
constexpr auto kClipByNormNoDivSumOpName = "ClipByNormNoDivSum";
constexpr auto kGreaterOpName = "Greater";
constexpr auto kSqrtOpName = "Sqrt";
constexpr auto kRsqrtOpName = "Rsqrt";
constexpr auto kErfOpName = "Erf";
constexpr auto kRealDivOpName = "RealDiv";
constexpr auto kLambUpdateWithLROpName = "LambUpdateWithLR";
constexpr auto kLambNextMVWithDecayOpName = "LambNextMVWithDecay";
constexpr auto kLambNextMVWithDecayV1OpName = "LambNextMVWithDecayV1";
constexpr auto kClipByValueOpName = "ClipByValue";
constexpr auto kLambNextRightOpName = "LambNextRight";
constexpr auto kConfusionSoftmaxGradOpName = "ConfusionSoftmaxGrad";
constexpr auto kLambUpdateWithLrV2OpName = "LambUpdateWithLrV2";
constexpr auto kLayerNormXBackpropOpName = "LayerNormXBackprop";
constexpr auto kLayerNormBetaGammaBackpropOpName = "LayerNormBetaGammaBackprop";
constexpr auto kLambNextMVOpName = "LambNextMV";
constexpr auto kConfusionTransposeDOpName = "ConfusionTransposeD";
constexpr auto kAdamApplyOneWithDecayOpName = "AdamApplyOneWithDecay";
constexpr auto kAdamApplyOneWithDecayAssignOpName = "AdamApplyOneWithDecayAssign";
constexpr auto kBatchNormGradOpName = "BatchNormGrad";
constexpr auto kBNInferOpName = "BNInfer";
constexpr auto kAdamApplyOneOpName = "AdamApplyOne";
constexpr auto kAdamApplyOneAssignOpName = "AdamApplyOneAssign";
constexpr auto kResizeNearestNeighborGradOpName = "ResizeNearestNeighborGrad";
constexpr auto kFusedMulAddOpName = "FusedMulAdd";
constexpr auto kFusedMulAddNOpName = "FusedMulAddN";
constexpr auto kFusedMulApplyMomentumOpName = "FusedMulApplyMomentum";
constexpr auto kBiasAddOpName = "BiasAdd";
constexpr auto kConfusionMulGradOpName = "ConfusionMulGrad";
constexpr auto kStreamSwitchOpName = "StreamSwitch";
constexpr auto kStreamActiveOpName = "StreamActive";
constexpr auto kAssignAddOpName = "AssignAdd";
constexpr auto kSendOpName = "Send";
constexpr auto kRecvOpName = "Recv";
constexpr auto kReluV2OpName = "ReLUV2";
constexpr auto kReluGradV2OpName = "ReluGradV2";
constexpr auto kAddNOpName = "AddN";
constexpr auto kResizeNearestNeighborV2OpName = "ResizeNearestNeighborV2";
constexpr auto kResizeNearestNeighborV2GradOpName = "ResizeNearestNeighborV2Grad";
constexpr auto kApplyRMSPropOpname = "ApplyRMSProp";
constexpr auto kCumsumOpName = "Cumsum";
constexpr auto kInplaceAddOpName = "InplaceAdd";
constexpr auto kInplaceSubOpName = "InplaceSub";
constexpr auto kResizeBilinearV2OpName = "kResizeBilinearV2";
constexpr auto kReduceProdOpName = "ReduceProd";
constexpr auto kCumprodOpName = "Cumprod";
constexpr auto kSpaceToBatchOpName = "SpaceToBatch";
constexpr auto kBatchToSpaceOpName = "BatchToSpace";
constexpr auto kPadOpName = "Pad";
constexpr auto kConv2DBackpropInputOpName = "Conv2DBackpropInput";
constexpr auto kConv2DBackpropFilterOpName = "Conv2DBackpropFilter";
constexpr auto kDepthwiseConv2dNativeName = "DepthwiseConv2dNative";
constexpr auto kFusionOpConv2DBackpropInputReluGradV2Name = "FusionOp_Conv2DBackpropInput_ReluGradV2";
constexpr auto kFusionOpConv2DBackpropInputAddNReluGradV2Name = "FusionOp_Conv2DBackpropInput_AddN_ReluGradV2";
constexpr auto kLabelSetOpName = "LabelSet";
constexpr auto kLabelSwitchOpName = "LabelSwitch";
constexpr auto kLabelGotoOpName = "LabelGoto";
constexpr auto kBNInferGradOpName = "BNInferGrad";
constexpr auto kCallOpName = "call";
constexpr auto kPartialOpName = "partial";
constexpr auto kSwitchOpName = "switch";
constexpr auto kReturnOpName = "return";
constexpr auto kLarsV2OpName = "LarsV2";
constexpr auto kLarsV2UpdateOpName = "LarsV2Update";
constexpr auto kSquareSumAllOpName = "SquareSumAll";
constexpr auto kNMSWithMaskOpName = "NMSWithMask";
constexpr auto kSoftmaxGradExtOpName = "SoftmaxGradExt";
constexpr auto kStridedReadOpName = "StridedRead";
constexpr auto kStridedWriteOpName = "StridedWrite";
constexpr auto kFusedAdamWeightDecayName = "FusedAdamWeightDecay";
constexpr auto kFusedAdamName = "FusedAdam";
constexpr auto kFusedSparseAdamName = "FusedSparseAdam";
constexpr auto kApplyAdagradV2OpName = "ApplyAdagradV2";
constexpr auto kSparseApplyAdagradV2OpName = "SparseApplyAdagradV2";
constexpr auto kSparseApplyFtrlOpName = "SparseApplyFtrl";
constexpr auto kSparseApplyFtrlV2OpName = "SparseApplyFtrlV2";
constexpr auto kApplyKerasMomentumOpName = "ApplyKerasMomentum";
constexpr auto kSparseApplyProximalAdagradOpName = "SparseApplyProximalAdagrad";
constexpr auto kSparseApplyRMSPropOpName = "SparseApplyRMSProp";
constexpr auto kSparseApplyAdadeltaOpName = "SparseApplyAdadelta";
constexpr auto kApplyAdamWithAmsgradOpName = "ApplyAdamWithAmsgrad";
constexpr auto kTensorMoveOpName = "TensorMove";
constexpr auto kTensorScatterUpdateOpName = "TensorScatterUpdate";
constexpr auto kScatterNdUpdateOpName = "ScatterNdUpdate";
constexpr auto kPushOpName = "Push";
constexpr auto kPullOpName = "Pull";
constexpr auto kEmbeddingLookupOpName = "EmbeddingLookup";
constexpr auto kEmbeddingLookupProxyOpName = "EmbeddingLookupProxy";
constexpr auto kPaddingOpName = "Padding";
constexpr auto kAvgPoolOpName = "AvgPool";
constexpr auto kAvgPoolGradGpuOpName = "AvgPoolGradGpu";
constexpr auto kmaxPoolGradOpName = "MaxPoolGrad";
constexpr auto kTensorAddOpName = "TensorAdd";
constexpr auto kCastOpName = "Cast";
constexpr auto kGreaterEqualOpName = "GreaterEqual";
constexpr auto kAbsOpName = "Abs";
constexpr auto kExpOpName = "Exp";
constexpr auto kNegOpName = "Neg";
constexpr auto kMinimumOpName = "Minimum";
constexpr auto kMaximumOpName = "Maximum";
constexpr auto kMulOpName = "Mul";
constexpr auto kSubOpName = "Sub";
constexpr auto kLogOpName = "Log";
constexpr auto kPowOpName = "Pow";
constexpr auto kReciprocalOpName = "Reciprocal";
constexpr auto kEqualOpName = "Equal";
constexpr auto kLessOpName = "Less";
constexpr auto kLessEqualOpName = "LessEqual";
constexpr auto kSquareOpName = "Square";
constexpr auto kSelectOpName = "Select";
constexpr auto kReduceSumOpName = "ReduceSum";
constexpr auto kReduceMinOpName = "ReduceMin";
constexpr auto kReduceMaxOpName = "ReduceMax";
constexpr auto kFusedWeightScaleApplyMomentum = "FusedWeightScaleApplyMomentum";
constexpr auto kFusedScaleApplyMomentum = "FusedScaleApplyMomentum";
constexpr auto kBasicLSTMCellWeightGradOpName = "BasicLSTMCellWeightGrad";
constexpr auto kBasicLSTMCellInputGradOpName = "BasicLSTMCellInputGrad";
constexpr auto kBasicLSTMCellOpName = "BasicLSTMCell";
constexpr auto kDynamicRNNOpName = "DynamicRNN";
constexpr auto kLSTMInputGradOpName = "LSTMInputGrad";
constexpr auto kDynamicGRUV2OpName = "DynamicGRUV2";
constexpr auto kGRUV2HiddenGradOpName = "GRUV2HiddenGrad";
constexpr auto kFusedSparseFtrlName = "FusedSparseFtrl";
constexpr auto kFusedSparseProximalAdagradName = "FusedSparseProximalAdagrad";
constexpr auto kFusedSparseLazyAdamName = "FusedSparseLazyAdam";
constexpr auto kSparseApplyFtrlName = "SparseApplyFtrl";
constexpr auto kSparseApplyFtrlV2Name = "SparseApplyFtrlV2";
constexpr auto kSGDName = "SGD";
constexpr auto kLARSUpdateName = "LARSUpdate";
constexpr auto kBasicLSTMCellCStateGradOpName = "BasicLSTMCellCStateGrad";
constexpr auto kBasicLSTMCellCStateGradV2OpName = "BasicLSTMCellCStateGradV2";
constexpr auto kMatMulV2OpName = "MatMulV2";
constexpr auto kBroadcastToOpName = "BroadcastTo";

// Hcom Op Type
constexpr auto kHcomOpTypeAllReduce = "HcomAllReduce";
constexpr auto kHcomOpTypeAllGather = "HcomAllGather";
constexpr auto kHcomOpTypeBroadcast = "HcomBroadcast";
constexpr auto kHcomOpTypeReduceScatter = "HcomReduceScatter";

// attr key name
constexpr auto kAttrInputNames = "input_names";
constexpr auto kAttrIsAICPUKernel = "is_AICPU_kernel";
constexpr auto kIsBackendCast = "is_backed_cast";
constexpr auto kAttrOutputNames = "output_names";
constexpr auto kAttrVisited = "visited";
constexpr auto kAttrShape = "shape";
constexpr auto kAttrMomentum = "momentum";
constexpr auto kAttrEps = "eps";
constexpr auto kAttrEpsilon = "epsilon";
constexpr auto kAttrFactor = "factor";
constexpr auto kAttrIsRef = "isRef";
constexpr auto kAttrDataShape = "data_shape";
constexpr auto kAttrAxis = "axis";
constexpr auto kAttrKeepDims = "keep_dims";
constexpr auto kAttrShapeGamma = "shape_gamma";
constexpr auto kAttrPerm = "perm";
constexpr auto kAttrTransposeFirst = "transpose_first";
constexpr auto kAttrAtomicAddMemSize = "automic_add_mem_size";
constexpr auto kAttrAtomicOutputIndexs = "atomic_output_clean_indexs";
constexpr auto kAttrAtomicWorkspaceIndexs = "atomic_workspace_clean_indexs";
constexpr auto kAttrSwitchCondition = "switch_condition";
constexpr auto kAttrDataType = "data_type";
constexpr auto kAttrActiveTarget = "active_target";
constexpr auto kAttrActiveStreamList = "active_stream_list";
constexpr auto kAttrTrueBranchStream = "true_branch_stream";
constexpr auto kAttrStreamSwitchKind = "stream_switch_kind";
constexpr auto kAttrEventId = "event_id";
constexpr auto kAttrDynInput = "dynamic";
constexpr auto kAttrDynInputSizes = "dyn_input_sizes";
constexpr auto kAttrSrcFormat = "src_format";
constexpr auto kAttrDstFormat = "dst_format";
constexpr auto kAttrMultiples = "multiples";
constexpr auto kAttrFixPrecision = "fix_precision";
constexpr auto kAttrOutputPrecision = "output_precision";
constexpr auto kAttrOutputUsedNum = "output_used_num";
constexpr auto kAttrHasBias = "has_bias";
constexpr auto kAttrN = "n";
constexpr auto kAttrLabelForInsertStreamActive = "label_for_insert_stream_active";
constexpr auto kAttrFusion = "fusion";
constexpr auto kAttrGroup = "group";
constexpr auto kAttrOp = "op";
constexpr auto kAttrRootRank = "root_rank";
constexpr auto kAttrIsTraining = "is_training";
constexpr auto kAttrFusionId = "fusion_id";
constexpr auto kAttrLabelIndex = "label_index";
constexpr auto kAttrLabelSwitchList = "label_switch_list";
constexpr auto kAttrNewAxisMask = "new_axis_mask";
constexpr auto kAttrShrinkAxisMask = "shrink_axis_mask";
constexpr auto kAttrDatadumpOriginalNames = "_datadump_original_names";
constexpr auto kAttrDatadumpIsMultiop = "_datadump_is_multiop";
constexpr auto kAttrStreamId = "stream_id";
constexpr auto kAttrRecordEvent = "record_event";
constexpr auto kAttrWaitEvent = "wait_event";
constexpr auto kAttrRecordEventStream = "record_event_stream";
constexpr auto kAttrWaitEventStream = "wait_event_stream";
constexpr auto kAttrIndex = "index";
constexpr auto kAttrSplitDim = "split_dim";
constexpr auto kAttrNumSplit = "num_split";
constexpr auto kAttrOutputNum = "output_num";
constexpr auto kAttrSizeSplits = "size_splits";
constexpr auto kAttrOutputDefault = "output_default";
constexpr auto kAttrPrimitiveTarget = "primitive_target";
constexpr auto kAttrUseLocking = "use_locking";
constexpr auto kAttrReduceScatterFlag = "reduce_scatter_flag";
constexpr auto kAttrOffset = "offset";
constexpr auto kAttrPsKey = "ps_key";
constexpr auto kAttrOptimizerType = "optim_type";
constexpr auto kAttrChildGraph = "child_graph";
constexpr auto kAttrInputNums = "inputNums";
constexpr auto kAttrT = "T";
constexpr auto kAttrNum = "num";
constexpr auto kAttrRankSize = "rank_size";
constexpr auto kAttrPadDimSize = "pad_dim_size";
constexpr auto kAttrNumSegments = "num_segments";
constexpr auto kAttrBegin = "begin";
constexpr auto kAttrSize = "size";
constexpr auto kAttrIsDynamicShape = "is_dynamic_shape";
constexpr auto kAttrInputIsDynamicShape = "input_is_dynamic_shape";
constexpr auto kAttrOutputIsDynamicShape = "output_is_dynamic_shape";
constexpr auto kAttrCompileInfo = "compile_info";
constexpr auto kAttrFusionType = "fusion_type";

// attr value
constexpr auto kValueTargetSwitch = "target_switch";
constexpr auto kValueTargetOther = "target_other";

// some size
const size_t kShape4dDims = 4;
const size_t kShape2dDims = 2;
const size_t kShape5dDims = 5;
const size_t kShape1dDims = 1;
const size_t kCubeSize = 16;
const size_t kMemAlignSize = 512;
const int kParameterDataTensorMask = 0;
const int kParameterWeightTensorMask = 1;
const int kValueNodeTensorMask = 2;

// define special index in special node
constexpr auto kAnfPrimitiveIndex = 0;
constexpr auto kFirstDataInputIndex = 1;
constexpr auto kRealInputNodeIndexInTupleGetItem = 1;
constexpr auto kInputNodeOutputIndexInTupleGetItem = 2;
constexpr auto kTupleGetItemInputSize = 3;
constexpr auto kSwitchInputSize = 4;
constexpr auto kFirstBranchInSwitch = 2;
constexpr auto kCallKernelGraphIndex = 1;
constexpr auto kSwitchTrueKernelGraphIndex = 2;
constexpr auto kSwitchFalseKernelGraphIndex = 3;
// index define of control depend
constexpr auto kControlDependPriorIndex = 1;
constexpr auto kControlDependBehindIndex = 2;
constexpr auto kControlDependMode = "depend_mode";
// index define of depend
constexpr auto kRealInputIndexInDepend = 1;
constexpr auto kDependAttachNodeIndex = 2;
constexpr auto kDependInputSize = 3;
// format
constexpr auto kOpFormat_DEFAULT = "DefaultFormat";
constexpr auto kOpFormat_NC1KHKWHWC0 = "NC1KHKWHWC0";
constexpr auto kOpFormat_ND = "ND";
constexpr auto kOpFormat_NCHW = "NCHW";
constexpr auto kOpFormat_NHWC = "NHWC";
constexpr auto kOpFormat_HWCN = "HWCN";
constexpr auto kOpFormat_NC1HWC0 = "NC1HWC0";
constexpr auto kOpFormat_FRAC_Z = "FracZ";
constexpr auto kOpFormat_FRAC_NZ = "FRACTAL_NZ";
constexpr auto kOpFormat_C1HWNCoC0 = "C1HWNCoC0";
constexpr auto kOpFormat_NC1HWC0_C04 = "NC1HWC0_C04";
constexpr auto kOpFormat_FRACTAL_Z_C04 = "FRACTAL_Z_C04";
constexpr auto kOpFormat_NDHWC = "NDHWC";
constexpr auto kOpFormat_FRACTAL_ZN_LSTM = "FRACTAL_ZN_LSTM";
const std::set<std::string> kOpFormatList = {
  kOpFormat_DEFAULT,     kOpFormat_NC1KHKWHWC0,   kOpFormat_ND,     kOpFormat_NCHW,           kOpFormat_NHWC,
  kOpFormat_HWCN,        kOpFormat_NC1HWC0,       kOpFormat_FRAC_Z, kOpFormat_C1HWNCoC0,      kOpFormat_FRAC_NZ,
  kOpFormat_NC1HWC0_C04, kOpFormat_FRACTAL_Z_C04, kOpFormat_NDHWC,  kOpFormat_FRACTAL_ZN_LSTM};
const std::set<std::string> kDefaultCompatibleFormat = {kOpFormat_ND, kOpFormat_NCHW, kOpFormat_NHWC, kOpFormat_HWCN};
const std::set<std::string> kOptOperatorSet = {kMomentumOpName,
                                               kApplyMomentumOpName,
                                               kApplyAdadeltaOpName,
                                               kApplyAdagradOpName,
                                               kApplyAdagradDAName,
                                               kApplyAdamOpName,
                                               kApplyAdaMaxOpName,
                                               kApplyAddSignOpName,
                                               kApplyCenteredRMSPOpName,
                                               kApplyFtrlOpName,
                                               kApplyFtrlV2OpName,
                                               kApplyGradientDescentOpName,
                                               kApplyPowerSignOpName,
                                               kApplyProximalAdagradOpName,
                                               kApplyProximalGradientDescentOpName,
                                               kApplyRMSPropOpName,
                                               kFusedAdamWeightDecayName,
                                               kFusedAdamName,
                                               kFusedSparseAdamName,
                                               kFusedWeightScaleApplyMomentum,
                                               kFusedScaleApplyMomentum,
                                               kApplyCenteredRMSPropOpName,
                                               kFusedSparseFtrlName,
                                               kFusedSparseProximalAdagradName,
                                               kFusedSparseLazyAdamName,
                                               kSparseApplyFtrlName,
                                               kSparseApplyFtrlV2Name,
                                               kSGDName,
                                               kLARSUpdateName,
                                               kPullOpName,
                                               kCombineMomentumWeightOpName,
                                               kCombineMomentumOpName,
                                               kSparseApplyProximalAdagradOpName};

const std::set<std::string> kHWSpecialFormatSet = {
  kOpFormat_FRAC_Z,    kOpFormat_NC1KHKWHWC0, kOpFormat_NC1HWC0,       kOpFormat_FRAC_NZ,
  kOpFormat_C1HWNCoC0, kOpFormat_NC1HWC0_C04, kOpFormat_FRACTAL_Z_C04, kOpFormat_FRACTAL_ZN_LSTM};

const std::set<TypeId> kFloatDataTypeSet = {kNumberTypeFloat16, kNumberTypeFloat32};

static inline void ChangeFileMode(const std::string &file_name, mode_t mode) {
  try {
    if (chmod(file_name.c_str(), mode) != 0) {
      MS_LOG(DEBUG) << "Change file `" << file_name << "` to mode " << std::oct << mode << " fail.";
    }
  } catch (std::exception &e) {
    MS_LOG(DEBUG) << "File `" << file_name << "` change mode failed! May be not exist.";
  }
}

static inline uint64_t GetCurrentUSec() {
  struct timeval tv;
  int ret = gettimeofday(&tv, nullptr);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "Fail gettimeofday, ret = " << ret;
  }
  return static_cast<uint64_t>(tv.tv_usec + tv.tv_sec * 1000000);
}

#define PROF_START(stage) uint64_t start_usec_##stage = mindspore::GetCurrentUSec()
#define PROF_END(stage)                                                                         \
  do {                                                                                          \
    uint64_t end_usec_##stage = mindspore::GetCurrentUSec();                                    \
    MS_LOG(INFO) << #stage << " costs " << (end_usec_##stage - start_usec_##stage) << " usec."; \
  } while (0)

#define PROF_MULTI_DEFINE(stage)     \
  static uint64_t total_##stage = 0; \
  static uint64_t count_##stage = 0;

#define PROF_LOCAL_DEFINE(stage) \
  uint64_t total_##stage = 0;    \
  uint64_t count_##stage = 0;

#define PROF_MULTI_START(stage) uint64_t start_usec_##stage = mindspore::GetCurrentUSec()

#define PROF_MULTI_END(stage)                                 \
  do {                                                        \
    ++count_##stage;                                          \
    uint64_t end_usec_##stage = mindspore::GetCurrentUSec();  \
    total_##stage += (end_usec_##stage - start_usec_##stage); \
  } while (0)

#define PROF_MULTI_PRINT(stage)                                                                             \
  do {                                                                                                      \
    MS_LOG(INFO) << #stage << " called " << count_##stage << " times, costs " << total_##stage << " usec."; \
  } while (0)
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_UTILS_UTILS_H_

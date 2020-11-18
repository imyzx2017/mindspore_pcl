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
#include "src/runtime/kernel/arm/base/detection_post_process_base.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/int8/quant_dtype_cast_int8.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_DetectionPostProcess;

namespace mindspore::kernel {

void PartialArgSort(const float *scores, int *indexes, int num_to_sort, int num_values) {
  std::partial_sort(indexes, indexes + num_to_sort, indexes + num_values, [&scores](const int i, const int j) {
    if (scores[i] == scores[j]) {
      return i < j;
    }
    return scores[i] > scores[j];
  });
}

int DetectionPostProcessBaseCPUKernel::Init() {
  params_->decoded_boxes_ = nullptr;
  params_->nms_candidate_ = nullptr;
  params_->indexes_ = nullptr;
  params_->scores_ = nullptr;
  params_->all_class_indexes_ = nullptr;
  params_->all_class_scores_ = nullptr;
  params_->single_class_indexes_ = nullptr;
  params_->selected_ = nullptr;
  params_->anchors_ = nullptr;
  auto anchor_tensor = in_tensors_.at(2);
  if (anchor_tensor->data_type() == kNumberTypeInt8) {
    auto quant_param = anchor_tensor->GetQuantParams().front();
    auto anchor_int8 = reinterpret_cast<int8_t *>(anchor_tensor->data_c());
    MS_ASSERT(anchor_int8 != nullptr);
    auto anchor_fp32 = new (std::nothrow) float[anchor_tensor->ElementsNum()];
    if (anchor_fp32 == nullptr) {
      MS_LOG(ERROR) << "Malloc anchor failed";
      return RET_ERROR;
    }
    DoDequantizeInt8ToFp32(anchor_int8, anchor_fp32, quant_param.scale, quant_param.zeroPoint,
                           anchor_tensor->ElementsNum());
    params_->anchors_ = anchor_fp32;
  } else if (anchor_tensor->data_type() == kNumberTypeUInt8) {
    auto quant_param = anchor_tensor->GetQuantParams().front();
    auto anchor_uint8 = reinterpret_cast<uint8_t *>(anchor_tensor->data_c());
    MS_ASSERT(anchor_uint8 != nullptr);
    auto anchor_fp32 = new (std::nothrow) float[anchor_tensor->ElementsNum()];
    if (anchor_fp32 == nullptr) {
      MS_LOG(ERROR) << "Malloc anchor failed";
      return RET_ERROR;
    }
    DoDequantizeUInt8ToFp32(anchor_uint8, anchor_fp32, quant_param.scale, quant_param.zeroPoint,
                            anchor_tensor->ElementsNum());
    params_->anchors_ = anchor_fp32;
  } else if (anchor_tensor->data_type() == kNumberTypeFloat32 || anchor_tensor->data_type() == kNumberTypeFloat) {
    params_->anchors_ = new (std::nothrow) float[anchor_tensor->ElementsNum()];
    if (params_->anchors_ == nullptr) {
      MS_LOG(ERROR) << "Malloc anchor failed";
      return RET_ERROR;
    }
    memcpy(params_->anchors_, anchor_tensor->data_c(), anchor_tensor->Size());
  } else {
    MS_LOG(ERROR) << "unsupported anchor data type " << anchor_tensor->data_type();
    return RET_ERROR;
  }
  return RET_OK;
}

DetectionPostProcessBaseCPUKernel::~DetectionPostProcessBaseCPUKernel() { delete[](params_->anchors_); }

int DetectionPostProcessBaseCPUKernel::ReSize() { return RET_OK; }

int NmsMultiClassesFastCoreRun(void *cdata, int task_id) {
  auto KernelData = reinterpret_cast<DetectionPostProcessBaseCPUKernel *>(cdata);
  int ret = NmsMultiClassesFastCore(KernelData->num_boxes_, KernelData->num_classes_with_bg_, KernelData->input_scores_,
                                    PartialArgSort, KernelData->params_, task_id, KernelData->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "NmsMultiClassesFastCore error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

void DetectionPostProcessBaseCPUKernel::FreeAllocatedBuffer() {
  if (params_->decoded_boxes_ != nullptr) {
    context_->allocator->Free(params_->decoded_boxes_);
    params_->decoded_boxes_ = nullptr;
  }
  if (params_->nms_candidate_ != nullptr) {
    context_->allocator->Free(params_->nms_candidate_);
    params_->nms_candidate_ = nullptr;
  }
  if (params_->indexes_ != nullptr) {
    context_->allocator->Free(params_->indexes_);
    params_->indexes_ = nullptr;
  }
  if (params_->scores_ != nullptr) {
    context_->allocator->Free(params_->scores_);
    params_->scores_ = nullptr;
  }
  if (params_->all_class_indexes_ != nullptr) {
    context_->allocator->Free(params_->all_class_indexes_);
    params_->all_class_indexes_ = nullptr;
  }
  if (params_->all_class_scores_ != nullptr) {
    context_->allocator->Free(params_->all_class_scores_);
    params_->all_class_scores_ = nullptr;
  }
  if (params_->single_class_indexes_ != nullptr) {
    context_->allocator->Free(params_->single_class_indexes_);
    params_->single_class_indexes_ = nullptr;
  }
  if (params_->selected_ != nullptr) {
    context_->allocator->Free(params_->selected_);
    params_->selected_ = nullptr;
  }
}

int DetectionPostProcessBaseCPUKernel::Run() {
  MS_ASSERT(context_->allocator != nullptr);
  int status = GetInputData();
  if (status != RET_OK) {
    return status;
  }
  auto output_boxes = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());
  auto output_classes = reinterpret_cast<float *>(out_tensors_.at(1)->MutableData());
  auto output_scores = reinterpret_cast<float *>(out_tensors_.at(2)->MutableData());
  auto output_num = reinterpret_cast<float *>(out_tensors_.at(3)->MutableData());

  num_boxes_ = in_tensors_.at(0)->shape()[1];
  num_classes_with_bg_ = in_tensors_.at(1)->shape()[2];
  params_->decoded_boxes_ = context_->allocator->Malloc(num_boxes_ * 4 * sizeof(float));
  if (params_->decoded_boxes_ == nullptr) {
    MS_LOG(ERROR) << "malloc params->decoded_boxes_ failed.";
    FreeAllocatedBuffer();
    return RET_ERROR;
  }
  params_->nms_candidate_ = context_->allocator->Malloc(num_boxes_ * sizeof(uint8_t));
  if (params_->nms_candidate_ == nullptr) {
    MS_LOG(ERROR) << "malloc params->nms_candidate_ failed.";
    FreeAllocatedBuffer();
    return RET_ERROR;
  }
  params_->selected_ = context_->allocator->Malloc(num_boxes_ * sizeof(int));
  if (params_->selected_ == nullptr) {
    MS_LOG(ERROR) << "malloc params->selected_ failed.";
    FreeAllocatedBuffer();
    return RET_ERROR;
  }
  params_->single_class_indexes_ = context_->allocator->Malloc(num_boxes_ * sizeof(int));
  if (params_->single_class_indexes_ == nullptr) {
    MS_LOG(ERROR) << "malloc params->single_class_indexes_ failed.";
    FreeAllocatedBuffer();
    return RET_ERROR;
  }

  if (params_->use_regular_nms_) {
    params_->scores_ = context_->allocator->Malloc((num_boxes_ + params_->max_detections_) * sizeof(float));
    if (params_->scores_ == nullptr) {
      MS_LOG(ERROR) << "malloc params->scores_ failed";
      FreeAllocatedBuffer();
      return RET_ERROR;
    }
    params_->indexes_ = context_->allocator->Malloc((num_boxes_ + params_->max_detections_) * sizeof(int));
    if (params_->indexes_ == nullptr) {
      MS_LOG(ERROR) << "malloc params->indexes_ failed";
      FreeAllocatedBuffer();
      return RET_ERROR;
    }
    params_->all_class_scores_ = context_->allocator->Malloc((num_boxes_ + params_->max_detections_) * sizeof(float));
    if (params_->all_class_scores_ == nullptr) {
      MS_LOG(ERROR) << "malloc params->all_class_scores_ failed";
      FreeAllocatedBuffer();
      return RET_ERROR;
    }
    params_->all_class_indexes_ = context_->allocator->Malloc((num_boxes_ + params_->max_detections_) * sizeof(int));
    if (params_->all_class_indexes_ == nullptr) {
      MS_LOG(ERROR) << "malloc params->all_class_indexes_ failed";
      FreeAllocatedBuffer();
      return RET_ERROR;
    }
  } else {
    params_->scores_ = context_->allocator->Malloc(num_boxes_ * sizeof(float));
    if (params_->scores_ == nullptr) {
      MS_LOG(ERROR) << "malloc params->scores_ failed";
      FreeAllocatedBuffer();
      return RET_ERROR;
    }
    params_->indexes_ = context_->allocator->Malloc(num_boxes_ * params_->num_classes_ * sizeof(int));
    if (!params_->indexes_) {
      MS_LOG(ERROR) << "malloc params->indexes_ failed.";
      FreeAllocatedBuffer();
      return RET_ERROR;
    }
  }

  status = DecodeBoxes(num_boxes_, input_boxes_, params_->anchors_, params_);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "DecodeBoxes error";
    FreeAllocatedBuffer();
    return status;
  }

  if (params_->use_regular_nms_) {
    status = DetectionPostProcessRegular(num_boxes_, num_classes_with_bg_, input_scores_, output_boxes, output_classes,
                                         output_scores, output_num, PartialArgSort, params_);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "DetectionPostProcessRegular error error_code[" << status << "]";
      FreeAllocatedBuffer();
      return status;
    }
  } else {
    status = ParallelLaunch(this->context_->thread_pool_, NmsMultiClassesFastCoreRun, this, op_parameter_->thread_num_);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "NmsMultiClassesFastCoreRun error error_code[" << status << "]";
      FreeAllocatedBuffer();
      return status;
    }
    status = DetectionPostProcessFast(num_boxes_, num_classes_with_bg_, input_scores_,
                                      reinterpret_cast<float *>(params_->decoded_boxes_), output_boxes, output_classes,
                                      output_scores, output_num, PartialArgSort, params_);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "DetectionPostProcessFast error error_code[" << status << "]";
      FreeAllocatedBuffer();
      return status;
    }
  }
  FreeAllocatedBuffer();
  return RET_OK;
}  // namespace mindspore::kernel
}  // namespace mindspore::kernel

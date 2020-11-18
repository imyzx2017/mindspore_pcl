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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NCCL_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NCCL_GPU_KERNEL_H_

#include <nccl.h>
#include <dlfcn.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/kernel_constants.h"
#include "runtime/device/gpu/distribution/collective_init.h"

namespace mindspore {
namespace kernel {
enum NcclKernelType {
  NCCL_ALL_REDUCE = 0,
  NCCL_ALL_GATHER,
  NCCL_REDUCE_SCATTER,
  NCCL_BROADCAST,
  NCCL_INVALID_TYPE = 255
};
const std::map<std::string, NcclKernelType> kNcclTypeMap = {
  {"AllReduce", NCCL_ALL_REDUCE},
  {"AllGather", NCCL_ALL_GATHER},
  {"ReduceScatter", NCCL_REDUCE_SCATTER},
  {"Broadcast", NCCL_BROADCAST},
};

static std::map<std::string, ncclDataType_t> kNcclDtypeMap = {
  {"kNumberTypeFloat32", ncclFloat}, {"kNumberTypeFloat16", ncclHalf}, {"kNumberTypeInt32", ncclInt}};

typedef ncclResult_t (*AllReduce)(const void *, void *, size_t, ncclDataType_t, ncclRedOp_t, cudaStream_t,
                                  const std::string &);
typedef ncclResult_t (*AllGather)(const void *, void *, size_t, ncclDataType_t, cudaStream_t, const std::string &);
typedef ncclResult_t (*ReduceScatter)(const void *, void *, size_t, ncclDataType_t, ncclRedOp_t, cudaStream_t,
                                      const std::string &);
typedef ncclResult_t (*Broadcast)(const void *, void *, size_t, ncclDataType_t, int, cudaStream_t, const std::string &);

template <typename T>
class NcclGpuKernel : public GpuKernel {
 public:
  NcclGpuKernel()
      : nccl_kernel_type_(NCCL_INVALID_TYPE),
        nccl_reduce_type_(ncclSum),
        group_name_(""),
        input_size_(0),
        output_size_(0),
        root_(0),
        collective_handle_(nullptr),
        comm_stream_(nullptr) {}
  ~NcclGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);

    cudaStream_t stream = comm_stream_ ? comm_stream_ : reinterpret_cast<cudaStream_t>(stream_ptr);
    switch (nccl_kernel_type_) {
      case NCCL_ALL_REDUCE: {
        auto all_reduce_funcptr =
          reinterpret_cast<AllReduce>(dlsym(const_cast<void *>(collective_handle_), "AllReduce"));
        MS_EXCEPTION_IF_NULL(all_reduce_funcptr);
        CHECK_NCCL_RET_WITH_EXCEPT((*all_reduce_funcptr)(input_addr, output_addr, output_size_ / sizeof(T),
                                                         nccl_data_type_, nccl_reduce_type_, stream, group_name_),
                                   "ncclAllReduce failed");
        break;
      }
      case NCCL_ALL_GATHER: {
        auto all_gather_funcptr =
          reinterpret_cast<AllGather>(dlsym(const_cast<void *>(collective_handle_), "AllGather"));
        MS_EXCEPTION_IF_NULL(all_gather_funcptr);
        CHECK_NCCL_RET_WITH_EXCEPT(
          (*all_gather_funcptr)(input_addr, output_addr, input_size_ / sizeof(T), nccl_data_type_, stream, group_name_),
          "ncclAllGather failed");
        break;
      }
      case NCCL_REDUCE_SCATTER: {
        auto reduce_scatter_funcptr =
          reinterpret_cast<ReduceScatter>(dlsym(const_cast<void *>(collective_handle_), "ReduceScatter"));
        MS_EXCEPTION_IF_NULL(reduce_scatter_funcptr);
        CHECK_NCCL_RET_WITH_EXCEPT((*reduce_scatter_funcptr)(input_addr, output_addr, output_size_ / sizeof(T),
                                                             nccl_data_type_, nccl_reduce_type_, stream, group_name_),
                                   "ncclReduceScatter failed");
        break;
      }
      case NCCL_BROADCAST: {
        auto broadcast_funcptr =
          reinterpret_cast<Broadcast>(dlsym(const_cast<void *>(collective_handle_), "Broadcast"));
        MS_EXCEPTION_IF_NULL(broadcast_funcptr);
        for (int i = 0; i < SizeToInt(input_size_list_.size()); ++i) {
          input_addr = GetDeviceAddress<T>(inputs, i);
          output_addr = GetDeviceAddress<T>(outputs, i);
          CHECK_NCCL_RET_WITH_EXCEPT((*broadcast_funcptr)(input_addr, output_addr, output_size_list_[i] / sizeof(T),
                                                          nccl_data_type_, root_, stream, group_name_),
                                     "ncclBroadcast failed");
        }
        break;
      }
      default: {
        MS_LOG(EXCEPTION) << "Kernel type " << nccl_kernel_type_ << " is not supported.";
      }
    }
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    nccl_data_type_ = kNcclDtypeMap[TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0))];
    InferCommType(kernel_node);

    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    for (size_t i = 0; i < input_num; ++i) {
      auto shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, i);
      size_t size = sizeof(T);
      for (size_t j = 0; j < shape.size(); j++) {
        size *= IntToSize(shape[j]);
      }
      size_t aligned_size = (nccl_kernel_type_ != NCCL_ALL_REDUCE) ? size : AlignMemorySize(size);
      input_size_list_.push_back(aligned_size);
      input_size_ += aligned_size;
    }
    for (size_t i = 0; i < output_num; ++i) {
      auto shape = AnfAlgo::GetOutputInferShape(kernel_node, i);
      size_t size = sizeof(T);
      for (size_t j = 0; j < shape.size(); j++) {
        size *= IntToSize(shape[j]);
      }
      size_t aligned_size = (nccl_kernel_type_ != NCCL_ALL_REDUCE) ? size : AlignMemorySize(size);
      output_size_list_.push_back(aligned_size);
      output_size_ += aligned_size;
    }

    group_name_ = GetAttr<std::string>(kernel_node, kAttrGroup);
    MS_LOG(INFO) << AnfAlgo::GetCNodeName(kernel_node) << " for group " << group_name_;
    auto comm_stream_attr = AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("stream_id");
    if (comm_stream_attr) {
      comm_stream_ = reinterpret_cast<cudaStream_t>(GetValue<uintptr_t>(comm_stream_attr));
      MS_EXCEPTION_IF_NULL(comm_stream_);
    }

    collective_handle_ = device::gpu::CollectiveInitializer::instance().collective_handle();
    MS_EXCEPTION_IF_NULL(collective_handle_);
    return true;
  }

 protected:
  void InitSizeLists() override { return; }

 private:
  void InferCommType(const CNodePtr &kernel_node) {
    std::string kernel_name = AnfAlgo::GetCNodeName(kernel_node);
    auto iter = kNcclTypeMap.find(kernel_name);
    if (iter == kNcclTypeMap.end()) {
      MS_LOG(EXCEPTION) << "Kernel " << kernel_name << " is not supported.";
    } else {
      nccl_kernel_type_ = iter->second;
    }

    auto reduce_op = AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr(kAttrOp);
    if (reduce_op) {
      std::string type = GetValue<std::string>(reduce_op);
      if (type == "sum") {
        nccl_reduce_type_ = ncclSum;
      } else if (type == "max") {
        nccl_reduce_type_ = ncclMax;
      } else if (type == "min") {
        nccl_reduce_type_ = ncclMin;
      } else if (type == "prod") {
        nccl_reduce_type_ = ncclProd;
      } else {
        MS_LOG(EXCEPTION) << "Nccl reduce type " << type << " is not supported.";
      }
    }

    auto root_rank = AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr(kAttrRootRank);
    if (root_rank) {
      root_ = static_cast<int>(GetValue<int64_t>(root_rank));
    }
    return;
  }

  size_t AlignMemorySize(size_t size) const {
    if (size == 0) {
      return COMMUNICATION_MEM_ALIGN_SIZE;
    }
    return ((size + COMMUNICATION_MEM_ALIGN_SIZE - 1) / COMMUNICATION_MEM_ALIGN_SIZE) * COMMUNICATION_MEM_ALIGN_SIZE;
  }

  NcclKernelType nccl_kernel_type_;
  ncclRedOp_t nccl_reduce_type_;
  ncclDataType_t nccl_data_type_;
  std::string group_name_;
  size_t input_size_;
  size_t output_size_;
  int root_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  const void *collective_handle_;
  cudaStream_t comm_stream_;

  static const size_t COMMUNICATION_MEM_ALIGN_SIZE = 16;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NCCL_GPU_KERNEL_H_

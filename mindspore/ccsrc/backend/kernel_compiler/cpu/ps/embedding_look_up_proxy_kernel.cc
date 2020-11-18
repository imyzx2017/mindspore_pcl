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
#include "backend/kernel_compiler/cpu/ps/embedding_look_up_proxy_kernel.h"
#include <vector>
#include "ps/worker.h"

namespace mindspore {
namespace kernel {
namespace ps {
void EmbeddingLookUpProxyKernel::InitKernel(const CNodePtr &kernel_node) {
  EmbeddingLookUpCPUKernel::InitKernel(kernel_node);
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  auto indices_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  auto output_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  size_t axis = kShape2dDims - input_shape.size();
  for (auto dim : input_shape) {
    input_dims_ *= dim;
  }
  if (input_dims_ * sizeof(float) > INT_MAX) {
    MS_LOG(EXCEPTION) << "PS mode embedding lookup max embedding table size is " << INT_MAX << ", current shape "
                      << input_shape << " is too large.";
  }

  if (mindspore::ps::Util::IsRoleOfWorker()) {
    key_ = AnfAlgo::GetNodeAttr<size_t>(kernel_node, kAttrPsKey);
  }
  std::vector<size_t> keys{key_, key_, key_};
  std::vector<size_t> values;
  values.insert(values.end(), input_shape.begin(), input_shape.end());
  values.insert(values.end(), indices_shape.begin(), indices_shape.end());
  values.insert(values.end(), output_shape.begin(), output_shape.end());
  MS_LOG(INFO) << "Init embedding lookup proxy kernel, input shape:" << input_shape
               << ", indices_shape:" << indices_shape << ", output_shape:" << output_shape;
  std::vector<int64_t> lens{SizeToLong(input_shape.size()), SizeToLong(indices_shape.size()),
                            SizeToLong(output_shape.size())};
  if (mindspore::ps::Util::IsRoleOfWorker()) {
    mindspore::ps::worker.AddEmbeddingTable(key_, input_shape[axis]);
    mindspore::ps::worker.InitPSEmbeddingTable(keys, values, lens);
  }
}

bool EmbeddingLookUpProxyKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> & /*workspace*/,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.size() != 2) {
    MS_LOG(EXCEPTION) << "Inputs size is " << inputs.size() << ", but EmbeddingLookUpProxyKernel needs 2.";
  }
  if (outputs.size() != 1) {
    MS_LOG(EXCEPTION) << "Outputs size is " << outputs.size() << ", but EmbeddingLookUpProxyKernel needs 1.";
  }
  auto indices_addr = reinterpret_cast<int *>(inputs[1]->addr);
  auto output_addr = reinterpret_cast<float *>(outputs[0]->addr);
  size_t input_size = inputs[1]->size;
  size_t output_size = outputs[0]->size;

  size_t size = input_size / sizeof(float);
  ::ps::SArray<int> lookup_ids(size, 0);
  ::ps::SArray<int> lengths{size};
  ::ps::SArray<float> lookup_result(output_size / sizeof(float), 0);
  auto ret = memcpy_s(lookup_ids.data(), lookup_ids.size() * sizeof(int), indices_addr, input_size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "Lookup id memcpy failed.";
    return false;
  }
  mindspore::ps::worker.DoPSEmbeddingLookup({key_}, lookup_ids, lengths, &lookup_result,
                                            mindspore::ps::kEmbeddingLookupCmd);

  auto ret2 = memcpy_s(output_addr, outputs[0]->size, lookup_result.data(), output_size);
  if (ret2 != EOK) {
    MS_LOG(EXCEPTION) << "Lookup result memcpy failed.";
    return false;
  }
  return true;
}
}  // namespace ps
}  // namespace kernel
}  // namespace mindspore

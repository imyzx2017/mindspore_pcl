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

#include "tools/converter/quantizer/post_training_quantizer.h"
#include <dirent.h>
#include <sys/stat.h>
#include <future>
#include <map>
#include <memory>
#include <algorithm>
#include <unordered_map>
#include <functional>
#include <numeric>
#include <utility>
#include <string>
#include <thread>
#include <vector>
#include <fstream>
#include "schema/inner/model_generated.h"
#include "src/tensor.h"
#include "tools/anf_exporter/anf_exporter.h"
#include "tools/converter/quantizer/quant_cast.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "src/common/log_adapter.h"
#include "securec/include/securec.h"
#include "tools/common/tensor_util.h"
#include "src/common/file_utils.h"
#include "src/common/utils.h"

using std::string;
using std::vector;

namespace mindspore {
namespace lite {
namespace quant {
STATUS DivergInfo::RecordMaxValue(const std::vector<float> &datas) {
  for (float data : datas) {
    max = std::max(data, max);
    min = std::min(data, min);
  }
  return RET_OK;
}

STATUS DivergInfo::RecordMaxValueArray(const std::vector<float> &datas) {
  if (datas.empty()) {
    return RET_ERROR;
  }
  float max_num = datas.at(0);
  float min_num = datas.at(0);
  for (float data : datas) {
    max_num = std::max(data, max_num);
    min_num = std::min(data, min_num);
  }
  this->max_datas.emplace_back(max_num);
  this->min_datas.emplace_back(min_num);
  return RET_OK;
}

void DivergInfo::UpdateInterval() {
  auto max_value = std::max(fabs(this->max), fabs(this->min));
  this->interval = max_value / static_cast<float>(bin_num);
}

STATUS DivergInfo::UpdateHistogram(const std::vector<float> &data) {
  for (auto value : data) {
    if (value == 0) {
      continue;
    }
    int bin_index = std::min(static_cast<int>(std::fabs(value) / this->interval), bin_num - 1);
    this->histogram[bin_index]++;
  }
  return RET_OK;
}

void DivergInfo::DumpHistogram() {
  MS_LOG(INFO) << "Print node " << cnode->fullname_with_scope() << " histogram";
  for (float item : this->histogram) {
    std::cout << item << " ";
  }
  std::cout << std::endl;
}

STATUS DivergInfo::ComputeThreshold() {
  if (method_x == kMethodMaxMin) {
    this->best_T = std::max(fabs(this->max), fabs(this->min));
    MS_LOG(DEBUG) << "using MAX_MIN, T: " << this->best_T;
    return RET_OK;
  }

  if (method_x == kMethodOutlier && this->min_datas.size() > 0) {
    this->percent_result = OutlierMethod(min_datas, max_datas);
    this->best_T = std::max(std::fabs(percent_result.first), std::fabs(percent_result.second));
    return RET_OK;
  }

  const constexpr int quant_bint_nums = 128;
  int threshold = quant_bint_nums;
  float min_kl = FLT_MAX;
  float after_threshold_sum = std::accumulate(this->histogram.begin() + quant_bint_nums, this->histogram.end(), 0.0f);

  for (int i = quant_bint_nums; i < this->bin_num; ++i) {
    std::vector<float> quantized_histogram(quant_bint_nums, 0);
    std::vector<float> reference_histogram(this->histogram.begin(), this->histogram.begin() + i);
    std::vector<float> expanded_histogram(i, 0);
    reference_histogram[i - 1] += after_threshold_sum;
    after_threshold_sum -= this->histogram[i];

    const float bin_interval = static_cast<float>(i) / static_cast<float>(quant_bint_nums);

    // merge i bins to target bins
    for (int j = 0; j < quant_bint_nums; ++j) {
      const float start = j * bin_interval;
      const float end = start + bin_interval;
      const int left_upper = static_cast<int>(std::ceil(start));
      if (left_upper > start) {
        const double left_scale = left_upper - start;
        quantized_histogram[j] += left_scale * this->histogram[left_upper - 1];
      }
      const int right_lower = static_cast<int>(std::floor(end));
      if (right_lower < end) {
        const double right_scale = end - right_lower;
        quantized_histogram[j] += right_scale * this->histogram[right_lower];
      }
      std::for_each(this->histogram.begin() + left_upper, this->histogram.begin() + right_lower,
                    [&quantized_histogram, j](float item) { quantized_histogram[j] += item; });
    }
    // expand target bins to i bins in order to calculate KL with reference_histogram
    for (int j = 0; j < quant_bint_nums; ++j) {
      const float start = j * bin_interval;
      const float end = start + bin_interval;
      float count = 0;
      const int left_upper = static_cast<int>(std::ceil(start));
      float left_scale = 0.0f;
      if (left_upper > start) {
        left_scale = left_upper - start;
        if (this->histogram[left_upper - 1] != 0) {
          count += left_scale;
        }
      }
      const int right_lower = static_cast<int>(std::floor(end));
      double right_scale = 0.0f;
      if (right_lower < end) {
        right_scale = end - right_lower;
        if (this->histogram[right_lower] != 0) {
          count += right_scale;
        }
      }
      std::for_each(this->histogram.begin() + left_upper, this->histogram.begin() + right_lower, [&count](float item) {
        if (item != 0) {
          count += 1;
        }
      });
      if (count == 0) {
        continue;
      }
      const float average_num = quantized_histogram[j] / count;
      if (left_upper > start && this->histogram[left_upper - 1] != 0) {
        expanded_histogram[left_upper - 1] += average_num * left_scale;
      }
      if (right_lower < end && this->histogram[right_lower] != 0) {
        expanded_histogram[right_lower] += average_num * right_scale;
      }
      for (int k = left_upper; k < right_lower; ++k) {
        if (this->histogram[k] != 0) {
          expanded_histogram[k] += average_num;
        }
      }
    }
    auto KLDivergence = [](std::vector<float> p, std::vector<float> q) {
      auto sum = 0.0f;
      std::for_each(p.begin(), p.end(), [&sum](float item) { sum += item; });
      std::for_each(p.begin(), p.end(), [sum](float &item) { item /= sum; });
      sum = 0.0f;
      std::for_each(q.begin(), q.end(), [&sum](float item) { sum += item; });
      std::for_each(q.begin(), q.end(), [sum](float &item) { item /= sum; });

      float result = 0.0f;
      const int size = p.size();
      for (int i = 0; i < size; ++i) {
        if (p[i] != 0) {
          if (q[i] == 0) {
            result += 1.0f;
          } else {
            result += (p[i] * std::log((p[i]) / (q[i])));
          }
        }
      }
      return result;
    };
    const float kl = KLDivergence(reference_histogram, expanded_histogram);
    if (kl < min_kl) {
      min_kl = kl;
      threshold = i;
    }
  }
  this->best_T = (static_cast<float>(threshold) + 0.5f) * this->interval;
  MS_LOG(DEBUG) << cnode->fullname_with_scope() << " Best threshold bin index: " << threshold << " T: " << best_T
                << " max: " << std::max(fabs(this->max), fabs(this->min));
  return RET_OK;
}

std::pair<CNodePtr, float> DivergInfo::GetScale() {
  float max_value = this->best_T;
  float min_value = -max_value;

  if (this->method_x == kMethodOutlier) {
    min_value = percent_result.first;
    max_value = percent_result.second;
  }

  MS_ASSERT(quant_max - quant_min != 0);
  float scale = (max_value - min_value) / (quant_max - quant_min);
  this->scale_tmp = scale;
  MS_ASSERT(scale != 0);
  return std::make_pair(this->cnode, scale);
}

std::pair<CNodePtr, int32_t> DivergInfo::GetZeropoint() {
  int zero_point = 0;
  if (quant_min == 0 && quant_max == 255) {
    zero_point = 128;
  } else if (quant_min == -127 && quant_max == 127) {
    zero_point = 0;
  } else {
    MS_LOG(WARNING) << "unexpectd quant range, quant_min: " << quant_min << " quant_max: " << quant_max;
  }

  if (this->method_x == kMethodOutlier) {
    zero_point = std::round(quant_max - percent_result.second / scale_tmp);
  }
  return std::make_pair(this->cnode, zero_point);
}

std::unordered_map<CNodePtr, float> Calibrator::GetScale(
  std::unordered_map<std::string, std::unique_ptr<DivergInfo>> *diverg_info) {
  std::unordered_map<CNodePtr, float> result;
  for (auto &iter : *diverg_info) {
    DivergInfo *info = iter.second.get();
    auto item = info->GetScale();
    result.insert(item);
  }
  return result;
}

std::unordered_map<CNodePtr, int32_t> Calibrator::GetZeropoint(
  std::unordered_map<std::string, std::unique_ptr<DivergInfo>> *diverg_info) {
  std::unordered_map<CNodePtr, int32_t> result;
  for (auto &iter : *diverg_info) {
    DivergInfo *info = iter.second.get();
    auto zeropoint = info->GetZeropoint();
    result.insert(zeropoint);
  }
  return result;
}

std::map<CNodePtr, MaxMin> Calibrator::GetMinMax(
  std::unordered_map<std::string, std::unique_ptr<DivergInfo>> *diverg_info) {
  std::map<CNodePtr, MaxMin> result;
  for (auto &iter : *diverg_info) {
    DivergInfo *info = iter.second.get();
    mindspore::lite::quant::MaxMin input_maxmin{};
    input_maxmin.min = info->min;
    input_maxmin.max = info->max;
    result[info->cnode] = input_maxmin;
  }
  return result;
}

void Calibrator::Dump() {
  for (auto &kv : this->inputs_diverg_info_) {
    auto &infos = kv.second;
    for (auto &info : infos) {
      info->DumpHistogram();
    }
  }
}

std::unordered_map<std::string, std::vector<std::unique_ptr<DivergInfo>>> *Calibrator::GetInputDivergInfo() {
  return &this->inputs_diverg_info_;
}

std::unordered_map<std::string, std::vector<std::unique_ptr<DivergInfo>>> *Calibrator::GetOutputDivergInfo() {
  return &this->outputs_diverg_info_;
}

STATUS Calibrator::RecordMaxValue(const vector<float> &data, const std::unique_ptr<DivergInfo> &diverg_info) {
  diverg_info->RecordMaxValue(data);
  diverg_info->RecordMaxValueArray(data);
  return RET_OK;
}

STATUS Calibrator::ComputeThreshold() {
  for (auto &kv : this->outputs_diverg_info_) {
    auto &outputs_diverg_info = kv.second;
    for (auto &diverg_info : outputs_diverg_info) {
      diverg_info->ComputeThreshold();
    }
  }
  // node A's input may be node B's output, no need to re-compute the node A's input quant param which is the same as
  for (auto &kv : this->inputs_diverg_info_) {
    auto &input_infos = kv.second;
    for (size_t i = 0; i < input_infos.size(); i++) {
      auto cnode = input_infos[i]->cnode;

      bool already_computed = false;
      auto input = cnode->input(i + 1);
      if (input->isa<mindspore::CNode>()) {
        auto input_cnode = std::dynamic_pointer_cast<mindspore::CNode>(input);
        for (const auto &outputs_diverg_info : outputs_diverg_info_) {
          if (already_computed) {
            break;
          }
          for (const auto &output_diverg_info : outputs_diverg_info.second) {
            auto output_diverg_cnode = output_diverg_info->cnode;
            if (output_diverg_cnode == input_cnode) {
              if (NodePrimitiveType(input_cnode) != schema::PrimitiveType_TupleGetItem) {
                *(input_infos[i]) = *output_diverg_info;
                input_infos[i]->cnode = cnode;
                already_computed = true;
                break;
              }
            }
          }
        }
      }
      if (!already_computed) {
        input_infos[i]->ComputeThreshold();
      }
    }
  }
  return RET_OK;
}

STATUS Calibrator::UpdateDivergInverval(
  std::unordered_map<std::string, std::vector<std::unique_ptr<DivergInfo>>> *diverg_info) {
  for (auto &kv : *diverg_info) {
    for (auto &info : kv.second) {
      info->UpdateInterval();
    }
  }
  return RET_OK;
}

STATUS Calibrator::UpdateDataFrequency(const vector<float> &data, const std::unique_ptr<DivergInfo> &diverg_info) {
  diverg_info->UpdateHistogram(data);
  return RET_OK;
}

STATUS Calibrator::AddQuantizedOp(const CNodePtr &node) {
  if (node == nullptr) {
    MS_LOG(ERROR) << "To be quantized node is null";
    return RET_ERROR;
  }
  string node_name = node->fullname_with_scope();
  std::unique_ptr<DivergInfo> input_diverg =
    std::make_unique<DivergInfo>(node, kDefaultBinNumber, bit_num_, quant_max_, quant_min_, config_param_.method_x);
  std::unique_ptr<DivergInfo> output_diverg =
    std::make_unique<DivergInfo>(node, kDefaultBinNumber, bit_num_, quant_max_, quant_min_, config_param_.method_x);

  inputs_diverg_info_[node_name].push_back(std::move(input_diverg));
  outputs_diverg_info_[node_name].push_back(std::move(output_diverg));
  return RET_OK;
}

void Calibrator::AddImage(const string &file, size_t index) {
  if (index >= images_.size()) {
    MS_LOG(ERROR) << "images_ size: " << images_.size() << " but index: " << index;
    return;
  }
  auto exist = [](const string &file) {
    struct stat buf {};
    return stat(file.c_str(), &buf) == 0;
  };
  if (exist(file)) {
    this->images_[index].push_back(file);
  } else {
    MS_LOG(WARNING) << "invalid image file path: " << file;
  }
}

STATUS Calibrator::GenerateInputData(int input_index, int image_index, mindspore::tensor::MSTensor *tensor) const {
  string path = images_[input_index][image_index];
  MS_LOG(INFO) << "read image: " << path;
  size_t size;
  char *bin_buf = ReadFile(path.c_str(), &size);
  auto data = tensor->MutableData();
  if (data == nullptr) {
    MS_LOG(ERROR) << "Get tensor MutableData return nullptr";
    return RET_ERROR;
  }
  if (size != tensor->Size()) {
    MS_LOG(ERROR) << "the input data is not consistent with model input, file_size: " << size
                  << " input tensor size: " << tensor->Size();
    return RET_ERROR;
  }
  auto ret = memcpy_s(data, tensor->Size(), bin_buf, size);
  if (ret != EOK) {
    MS_LOG(ERROR) << "memcpy_s error: " << ret;
    delete[] bin_buf;
    return RET_ERROR;
  }
  delete[] bin_buf;
  return RET_OK;
}

STATUS Calibrator::CollectImages() {
  this->images_.resize(config_param_.image_paths.size());
  auto input_i = 0;
  bool multi_input = config_param_.image_paths.size() > 1;
  for (const auto &image_path : config_param_.image_paths) {
    DIR *root = opendir(image_path.c_str());
    if (root == nullptr) {
      MS_LOG(ERROR) << "invalid image path: " << image_path;
      return RET_PARAM_INVALID;
    }
    struct dirent *image_dir = readdir(root);
    size_t count = 0;
    while (image_dir != nullptr) {
      string file_name(image_dir->d_name);
      if (file_name != "." && file_name != "..") {
        const std::string file_path = image_path + "/" + file_name;
        if (multi_input || config_param_.batch_count == 0) {
          this->AddImage(file_path, input_i);
          count++;
        } else if (count < config_param_.batch_count) {
          this->AddImage(file_path, input_i);
          count++;
        } else {
          break;
        }
      }
      image_dir = readdir(root);
    }
    std::sort(images_[input_i].begin(), images_[input_i].end());
    if (config_param_.batch_count != 0 && config_param_.batch_count < images_[input_i].size()) {
      images_[input_i].resize(config_param_.batch_count);
    }
    closedir(root);
    input_i++;
  }
  return RET_OK;
}

STATUS Calibrator::ReadConfig() {
  if (config_path_.empty() || config_path_.length() > PATH_MAX) {
    MS_LOG(ERROR) << "invalid config path!";
    return RET_PARAM_INVALID;
  }
  // check whether config file path is valid
  char *resolved_path = new (std::nothrow) char[PATH_MAX]{0};
  if (resolved_path == nullptr) {
    MS_LOG(ERROR) << "New an object failed.";
    return RET_ERROR;
  }
#ifdef _WIN32
  if (_fullpath(resolved_path, config_path_.c_str(), 1024) != nullptr) {
    config_path_ = string(resolved_path);
  }
#else
  if (realpath(config_path_.c_str(), resolved_path) != nullptr) {
    config_path_ = string(resolved_path);
  }
#endif
  std::ifstream fs(config_path_.c_str(), std::ifstream::in);
  if (!fs.is_open()) {
    MS_LOG(ERROR) << "config proto file %s open failed: " << config_path_;
    delete[] resolved_path;
    return RET_PARAM_INVALID;
  }
  std::string line;
  while (std::getline(fs, line)) {
    auto index = line.find('=');
    if (index == std::string::npos) {
      MS_LOG(ERROR) << "the config file is invalid, can not find '=', please check";
      delete[] resolved_path;
      return RET_PARAM_INVALID;
    }
    auto key = line.substr(0, index);
    auto value = line.substr(index + 1);
    Trim(&key);
    Trim(&value);
    if (key == "image_path") {
      auto &raw_image_paths = value;
      auto ind = raw_image_paths.find(',');
      while (ind != std::string::npos) {
        auto image_path = raw_image_paths.substr(0, ind);
        Trim(&image_path);
        config_param_.image_paths.push_back(image_path);
        raw_image_paths = raw_image_paths.substr(ind + 1);
        Trim(&raw_image_paths);
        ind = raw_image_paths.find(',');
      }
      config_param_.image_paths.push_back(raw_image_paths);
    } else if (key == "batch_count") {
      config_param_.batch_count = std::stoul(value);
    } else if (key == "thread_num") {
      config_param_.thread_num = std::stoul(value);
    } else if (key == "method_x") {
      if (value != kMethodKL && value != kMethodMaxMin && value != kMethodOutlier) {
        MS_LOG(WARNING) << "unsupported method_x: " << value << ". Use default value.";
      } else {
        config_param_.method_x = value;
      }
    } else if (key == "bias_correction") {
      std::for_each(value.begin(), value.end(), ::tolower);
      if (value == "true") {
        config_param_.bias_correction = true;
      }
    } else {
      MS_LOG(WARNING) << "unsupported parameter";
    }
  }

  for (const auto &path : config_param_.image_paths) {
    MS_LOG(DEBUG) << "calibration data_path: " << path;
  }
  MS_LOG(DEBUG) << "batch_count: " << config_param_.batch_count << "  "
                << "method_x: " << config_param_.method_x << "  "
                << "thread_num: " << config_param_.thread_num << " "
                << "bias_correction: " << config_param_.bias_correction;

  delete[] resolved_path;
  fs.close();
  return RET_OK;
}

Calibrator::Calibrator(string path, size_t bit_num, int quant_max, int quant_min)
    : config_path_(std::move(path)), bit_num_(bit_num), quant_max_(quant_max), quant_min_(quant_min) {}

PostTrainingQuantizer::PostTrainingQuantizer(FuncGraphPtr graph, string path, int bit_num, TypeId target_type,
                                             bool per_channel)
    : Quantizer(std::move(graph)) {
  this->per_channel_ = per_channel;
  this->bit_num = bit_num;
  this->target_type_ = target_type;
  if (target_type == kNumberTypeInt8) {
    quant_max = (1 << (this->bit_num - 1)) - 1;  // 127
    quant_min = -quant_max;                      // -127
  } else if (target_type == kNumberTypeUInt8) {
    quant_max = (1 << this->bit_num) - 1;  // 255
    quant_min = 0;
  } else {
    MS_LOG(ERROR) << "unsupported quant value type: " << target_type;
  }
  calibrator_ = std::make_unique<Calibrator>(std::move(path), this->bit_num, quant_max, quant_min);
  if (calibrator_ == nullptr) {
    MS_LOG(ERROR) << "creat calibrator failed!";
    return;
  }
}

STATUS PostTrainingQuantizer::DoQuantInput(double scale, int32_t zeropoint, struct MaxMin *max_min,
                                           const std::shared_ptr<PrimitiveC> &lite_primitive) const {
  schema::QuantParamT quant_param;
  quant_param.scale = scale;
  quant_param.zeroPoint = zeropoint;
  quant_param.max = max_min->max;
  quant_param.min = max_min->min;
  quant_param.numBits = bit_num;
  quant_param.narrowRange = false;
  quant_param.inited = true;
  std::vector<schema::QuantParamT> quant_params = {quant_param};
  lite_primitive->AddInputQuantParam(quant_params);
  return RET_OK;
}

STATUS PostTrainingQuantizer::DoQuantOutput(double scale, int zeropoint, struct MaxMin *max_min,
                                            const std::shared_ptr<PrimitiveC> &lite_primitive) const {
  schema::QuantParamT quant_param;
  quant_param.scale = scale;
  quant_param.zeroPoint = zeropoint;
  quant_param.max = max_min->max;
  quant_param.min = max_min->min;
  quant_param.numBits = bit_num;
  quant_param.narrowRange = false;
  quant_param.inited = true;
  std::vector<schema::QuantParamT> quant_params = {quant_param};
  lite_primitive->AddOutputQuantParam(quant_params);
  return RET_OK;
}

STATUS PostTrainingQuantizer::DoWeightQuant(const AnfNodePtr &weight, std::shared_ptr<PrimitiveC> primitive_c,
                                            bool perchanel) const {
  // perlayer
  if (!weight->isa<Parameter>()) {
    MS_LOG(ERROR) << "not a parameter";
    return RET_PARAM_INVALID;
  }
  auto parameter = std::dynamic_pointer_cast<Parameter>(weight);
  if (parameter == nullptr) {
    MS_LOG(ERROR) << weight->fullname_with_scope() << " can not cast to Parameter";
    return RET_ERROR;
  }
  ParamValueLitePtr paramValue = std::dynamic_pointer_cast<ParamValueLite>(parameter->default_param());
  if (paramValue == nullptr) {
    MS_LOG(ERROR) << weight->fullname_with_scope() << " can not get value";
    return RET_ERROR;
  }
  auto status = QuantFilter<int8_t>(paramValue, std::move(primitive_c), QuantType_PostTraining, quant_max, quant_min,
                                    bit_num, perchanel);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "QuantFilter failed: " << status;
    return status;
  }
  // set dtype
  auto abstractBase = parameter->abstract();
  if (abstractBase == nullptr) {
    MS_LOG(ERROR) << "Abstract of parameter is nullptr, " << parameter->name();
    return RET_ERROR;
  }
  if (!utils::isa<abstract::AbstractTensorPtr>(abstractBase)) {
    MS_LOG(ERROR) << "Abstract of parameter should be anstract tensor, " << parameter->name();
    return RET_ERROR;
  }
  auto abstractTensor = utils::cast<abstract::AbstractTensorPtr>(abstractBase);
  abstractTensor->element()->set_type(TypeIdToType(kNumberTypeInt8));
  return RET_OK;
}

STATUS PostTrainingQuantizer::DoBiasQuant(const AnfNodePtr &bias, const std::shared_ptr<PrimitiveC> &primitive_c) {
  if (primitive_c == nullptr || bias == nullptr) {
    MS_LOG(ERROR) << "null pointer!";
    return RET_NULL_PTR;
  }

  auto bias_parameter_ptr = std::dynamic_pointer_cast<Parameter>(bias);
  auto bias_default_param = bias_parameter_ptr->default_param();
  auto bias_param = std::dynamic_pointer_cast<ParamValueLite>(bias_default_param);

  auto active_weight_quant_params = primitive_c->GetInputQuantParams();
  if (active_weight_quant_params.size() != 2) {
    MS_LOG(ERROR) << "unexpected active_weight_quant_params size: " << active_weight_quant_params.size();
    return RET_ERROR;
  }

  auto active_params = active_weight_quant_params[0];
  auto weight_params = active_weight_quant_params[1];

  vector<double> input_scales;
  vector<double> filter_scales;
  vector<double> bias_scales;
  size_t sizeX = active_params.size();
  for (size_t i = 0; i < sizeX; i++) {
    input_scales.emplace_back(active_params[i].scale);
  }
  size_t sizeY = weight_params.size();
  if (sizeX != sizeY) {
    if (sizeX > 1 && sizeY > 1) {
      MS_LOG(ERROR) << "input and filter's scale count cannot match!";
      return RET_ERROR;
    }
  }
  for (size_t i = 0; i < sizeY; i++) {
    filter_scales.emplace_back(weight_params[i].scale);
  }
  size_t size = std::max(sizeX, sizeY);
  for (size_t i = 0; i < size; i++) {
    auto scaleX = sizeX > 1 ? input_scales[i] : input_scales[0];
    auto scaleY = sizeY > 1 ? filter_scales[i] : filter_scales[0];
    bias_scales.push_back(scaleX * scaleY);
  }
  MS_ASSERT(!bias_scales.empty());
  size_t shape_size = bias_param->tensor_shape_size();

  // set bias quant param
  vector<schema::QuantParamT> quant_params;
  for (double bias_scale : bias_scales) {
    schema::QuantParamT quant_param;
    quant_param.scale = bias_scale;
    quant_param.zeroPoint = 0;
    quant_param.inited = true;
    quant_params.emplace_back(quant_param);
  }
  // quant bias data
  std::vector<int32_t> quant_datas(shape_size);

  auto *raw_datas = static_cast<float *>(bias_param->tensor_addr());
  double bias_scale_tmp;
  const constexpr int32_t quanted_bias_abs_limit = 0.5 * INT32_MAX;

  if (bias_scales.size() == shape_size) {
    for (size_t i = 0; i < shape_size; i++) {
      bias_scale_tmp = bias_scales[i];
      if (std::abs(raw_datas[i] / bias_scale_tmp) >= quanted_bias_abs_limit) {
        MS_LOG(DEBUG) << "quanted bias over flow, maybe the scale of weight: " << active_weight_quant_params[1][i].scale
                      << " is too small, need to update";
        // update filter scale and zp
        double activate_scale = input_scales[0];
        double filter_scale = std::abs(raw_datas[i]) / (activate_scale * quanted_bias_abs_limit);
        active_weight_quant_params[1][i].scale = filter_scale;
        active_weight_quant_params[1][i].zeroPoint = 0;
        primitive_c->SetInputQuantParams(active_weight_quant_params);
        bias_scale_tmp = std::abs(raw_datas[i]) / quanted_bias_abs_limit;
        quant_params[i].scale = bias_scale_tmp;
        MS_LOG(DEBUG) << "new filter scale: " << filter_scale;
      }
      auto quant_data = (int32_t)std::round(raw_datas[i] / bias_scale_tmp);
      quant_datas[i] = quant_data;
    }
  } else if (bias_scales.size() == 1) {
    // for fc, per tensor quant
    bias_scale_tmp = quant_params[0].scale;
    float max_raw_data = 0.0f;
    for (size_t i = 0; i < shape_size; i++) {
      if (std::abs(raw_datas[i]) > max_raw_data) {
        max_raw_data = std::abs(raw_datas[i]);
      }
    }
    if (std::abs(max_raw_data / bias_scale_tmp) >= quanted_bias_abs_limit) {
      MS_LOG(DEBUG) << "quanted bias over flow, maybe the scale of weight: " << active_weight_quant_params[1][0].scale
                    << " is too small, need to update";
      double activate_scale = input_scales[0];
      double filter_scale = std::abs(max_raw_data) / (activate_scale * quanted_bias_abs_limit);
      active_weight_quant_params[1][0].scale = filter_scale;
      active_weight_quant_params[1][0].zeroPoint = 0;
      primitive_c->SetInputQuantParams(active_weight_quant_params);
      bias_scale_tmp = max_raw_data / quanted_bias_abs_limit;
      quant_params[0].scale = bias_scale_tmp;
      MS_LOG(DEBUG) << "new filter scale: " << filter_scale;
    }
    for (size_t i = 0; i < shape_size; i++) {
      auto quant_data = (int32_t)std::round(raw_datas[i] / bias_scale_tmp);
      quant_datas[i] = quant_data;
    }
  } else {
    MS_LOG(ERROR) << "unexpected input_scales size: " << input_scales.size()
                  << " weight_scales size: " << active_weight_quant_params[1].size();
    return RET_ERROR;
  }

  primitive_c->AddInputQuantParam(quant_params);
  auto ret =
    memcpy_s(bias_param->tensor_addr(), bias_param->tensor_size(), quant_datas.data(), shape_size * sizeof(int32_t));
  if (ret != EOK) {
    MS_LOG(ERROR) << "memcpy_s failed.";
    return RET_ERROR;
  }
  // set dtype
  auto abstractBase = bias_parameter_ptr->abstract();
  if (abstractBase == nullptr) {
    MS_LOG(ERROR) << "Abstract of parameter is nullptr, " << bias_parameter_ptr->name();
    return RET_ERROR;
  }
  if (!utils::isa<abstract::AbstractTensorPtr>(abstractBase)) {
    MS_LOG(ERROR) << "Abstract of parameter should be anstract tensor, " << bias_parameter_ptr->name();
    return RET_ERROR;
  }
  auto abstractTensor = utils::cast<abstract::AbstractTensorPtr>(abstractBase);
  abstractTensor->element()->set_type(TypeIdToType(kNumberTypeInt32));
  return RET_OK;
}

STATUS PostTrainingQuantizer::QuantNode() {
  auto inputs_diverg_info = calibrator_->GetInputDivergInfo();
  auto outputs_diverg_info = calibrator_->GetOutputDivergInfo();

  auto cnodes = funcGraph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    auto op_name = cnode->fullname_with_scope();
    if (this->calibrator_->GetInputDivergInfo()->find(op_name) == this->calibrator_->GetInputDivergInfo()->end()) {
      MS_LOG(INFO) << op_name << " can not do quant";
      continue;
    }
    auto primitive_c = GetValueNode<std::shared_ptr<PrimitiveC>>(cnode->input(0));
    if (primitive_c == nullptr) {
      MS_LOG(ERROR) << "primitive_c is nullptr";
      continue;
    }
    if (inputs_diverg_info->find(op_name) == inputs_diverg_info->end()) {
      primitive_c->SetQuantType(schema::QuantType_QUANT_NONE);
      continue;
    }

    auto op_type = (schema::PrimitiveType)primitive_c->Type();
    MS_LOG(DEBUG) << "OpName: " << op_name;
    if (op_type == PrimitiveType_TupleGetItem) {
      auto index_node = cnode->input(2);
      auto index_value_node = std::dynamic_pointer_cast<mindspore::ValueNode>(index_node);
      if (index_value_node == nullptr) {
        MS_LOG(WARNING) << "index value node is null";
        continue;
      }
      size_t index = GetValue<int>(index_value_node->value());

      auto input_node = cnode->input(1);
      auto input_cnode = std::dynamic_pointer_cast<mindspore::CNode>(input_node);
      auto input_cnode_primitive_c = GetValueNode<std::shared_ptr<PrimitiveC>>(input_cnode->input(0));
      if (input_cnode_primitive_c == nullptr) {
        MS_LOG(WARNING) << "input_cnode_primitive_c is null";
        continue;
      }
      if (input_cnode_primitive_c->GetOutputQuantParams().size() > index) {
        auto quant_param = input_cnode_primitive_c->GetOutputQuantParams()[index];
        primitive_c->AddInputQuantParam(quant_param);
        primitive_c->AddOutputQuantParam(quant_param);
      } else {
        MS_LOG(WARNING) << "this TupleGetItem node's input node: " << input_cnode->fullname_with_scope()
                        << "'s output quant_params size: " << input_cnode_primitive_c->GetOutputQuantParams().size()
                        << ", but index: " << index;
      }
      primitive_c->SetQuantType(schema::QuantType_PostTraining);
      continue;
    } else if (op_type != PrimitiveType_Conv2D && op_type != PrimitiveType_DepthwiseConv2D &&
               op_type != PrimitiveType_DeConv2D && op_type != PrimitiveType_DeDepthwiseConv2D &&
               op_type != PrimitiveType_FullConnection && op_type != PrimitiveType_LayerNorm) {
      for (size_t i = 1; i < cnode->inputs().size(); i++) {
        auto input_node = cnode->input(i);
        bool is_graph_input = false;
        if (input_node->isa<Parameter>()) {
          if (!input_node->cast<ParameterPtr>()->has_default()) {
            is_graph_input = true;
          }
        }
        if (input_node->isa<mindspore::CNode>()) {
          auto input_cnode = std::dynamic_pointer_cast<mindspore::CNode>(input_node);
          auto input_cnode_primitive_c = GetValueNode<std::shared_ptr<PrimitiveC>>(input_cnode->input(0));
          if (input_cnode_primitive_c == nullptr) {
            MS_LOG(DEBUG) << "input: " << i << " " << input_cnode->fullname_with_scope() << ": "
                          << " PrimitiveC is null";
            continue;
          }
          if (input_cnode_primitive_c->IsOutputQuantParamsInited()) {
            auto quant_param = input_cnode_primitive_c->GetOutputQuantParams().front();
            primitive_c->AddInputQuantParam(quant_param);
          } else {
            // do input quant
            auto &info = (*inputs_diverg_info)[op_name][i - 1];
            auto input_scale = info->GetScale().second;
            auto input_zp = info->GetZeropoint().second;
            struct MaxMin input_min_max {};
            input_min_max.max = info->max;
            input_min_max.min = info->min;
            DoQuantInput(input_scale, input_zp, &input_min_max, primitive_c);
          }
        } else if (is_graph_input) {
          auto &info = (*inputs_diverg_info)[op_name][i - 1];
          auto input_scale = info->GetScale().second;
          auto input_zp = info->GetZeropoint().second;
          struct MaxMin input_min_max {};
          input_min_max.max = info->max;
          input_min_max.min = info->min;
          DoQuantInput(input_scale, input_zp, &input_min_max, primitive_c);
        } else {
          MS_LOG(DEBUG) << "node: " << op_name << " input " << i << " not a cnode";
          // get dtype
          auto abstractBase = input_node->abstract();
          if (abstractBase == nullptr) {
            MS_LOG(ERROR) << "Abstract of parameter is nullptr, " << input_node->fullname_with_scope();
            return RET_ERROR;
          }
          if (!utils::isa<abstract::AbstractTensorPtr>(abstractBase)) {
            MS_LOG(ERROR) << "Abstract of parameter should be anstract tensor, " << input_node->fullname_with_scope();
            return RET_ERROR;
          }
          auto abstractTensor = utils::cast<abstract::AbstractTensorPtr>(abstractBase);
          if (abstractTensor->element()->GetTypeTrack()->type_id() == kNumberTypeFloat32) {
            MS_LOG(DEBUG) << "this parameter do quant";
            DoWeightQuant(input_node, primitive_c, false);
          } else {
            MS_LOG(DEBUG) << "this parameter no need to do quant";
          }
        }
      }
    } else {
      // do input quant
      auto &info = (*inputs_diverg_info)[op_name][0];
      auto input_scale = info->GetScale().second;
      auto input_zp = info->GetZeropoint().second;
      struct MaxMin input_min_max {};
      input_min_max.max = info->max;
      input_min_max.min = info->min;
      DoQuantInput(input_scale, input_zp, &input_min_max, primitive_c);
      // do weight quant
      auto weight = cnode->input(2);
      bool perchannel = false;
      if (op_type == PrimitiveType_Conv2D || op_type == PrimitiveType_DepthwiseConv2D) {
        perchannel = true;
      }
      DoWeightQuant(weight, primitive_c, perchannel);
      // do bias quant
      if (cnode->inputs().size() == 4) {
        auto bias = cnode->input(3);
        DoBiasQuant(bias, primitive_c);
      }
    }
    // do output quant, there may multi-output
    auto &infos = (*outputs_diverg_info)[op_name];
    for (auto &info : infos) {
      auto output_scale = info->GetScale().second;
      auto output_zp = info->GetZeropoint().second;
      struct MaxMin output_min_max {};
      output_min_max.max = info->max;
      output_min_max.min = info->min;

      DoQuantOutput(output_scale, output_zp, &output_min_max, primitive_c);
      primitive_c->SetQuantType(schema::QuantType_PostTraining);
    }
  }
  return RET_OK;
}

STATUS PostTrainingQuantizer::UpdateDivergInverval() {
  this->calibrator_->UpdateDivergInverval(this->calibrator_->GetInputDivergInfo());
  this->calibrator_->UpdateDivergInverval(this->calibrator_->GetOutputDivergInfo());
  return RET_OK;
}

/**
 * Pre Process
 * 1. generate config param
 *   1.1 read config file
 *   1.2 parse txt
 * 2. collect image files
 *   2.1 parse image files to input tensor
 * 3. save quantied node
 **/
STATUS PostTrainingQuantizer::PreProcess() {
  if (this->calibrator_ == nullptr) {
    MS_LOG(ERROR) << "calibrator is null!";
    return RET_ERROR;
  }
  // 1. generate config param
  STATUS status = calibrator_->ReadConfig();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "read proto text failed!";
    return status;
  }
  // 2. collect image files
  status = calibrator_->CollectImages();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "collect images failed!";
    return status;
  }
  // 3. collect to be quantized operators
  // from user input
  QuantStrategy strategy(10);
  auto cnodes = funcGraph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    AnfNodePtr anf = std::dynamic_pointer_cast<AnfNode>(cnode);
    if (strategy.CanOpPostQuantized(anf)) {
      calibrator_->AddQuantizedOp(cnode);
    }
    auto primitive_c = GetValueNode<std::shared_ptr<PrimitiveC>>(cnode->input(0));
    if (primitive_c == nullptr) {
      MS_LOG(ERROR) << cnode->fullname_with_scope() << " primitive is null";
      continue;
    }
    primitive_c->ClearInputOutputQuantParam();
  }
  return RET_OK;
}

STATUS PostTrainingQuantizer::CheckFp32TensorVec(const std::string &node_name,
                                                 const std::vector<mindspore::tensor::MSTensor *> &tensor_vec) const {
  if (tensor_vec.empty()) {
    MS_LOG(ERROR) << "node: " << node_name << " input tensors is 0";
    return RET_ERROR;
  }
  auto *tensor = tensor_vec[0];
  if (tensor->data_type() != kNumberTypeFloat32) {
    MS_LOG(WARNING) << "node: " << node_name << " will not quantize"
                    << " tensor data_type: " << tensor->data_type();
    return RET_ERROR;
  }
  return RET_OK;
}

/**
 * 1. create input tensor
 * 2. insert callback to session
 * 3. run session
 **/
STATUS PostTrainingQuantizer::DoInference() {
  // get input tensor
  vector<mindspore::tensor::MSTensor *> inputs = fp32_session_->GetInputs();
  if (inputs.size() != calibrator_->GetInputNum()) {
    MS_LOG(ERROR) << "model's input tensor cnt: " << inputs.size() << " != " << calibrator_->GetInputNum();
    return RET_ERROR;
  }

  for (size_t i = 0; i < calibrator_->GetBatchNum(); i++) {
    // set multi-input data
    for (size_t input_index = 0; input_index < inputs.size(); input_index++) {
      STATUS status = calibrator_->GenerateInputData(input_index, i, inputs[input_index]);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "generate input data from images failed!";
        return RET_ERROR;
      }
    }

    KernelCallBack beforeCallBack = [&](const std::vector<mindspore::tensor::MSTensor *> &beforeInputs,
                                        const std::vector<mindspore::tensor::MSTensor *> &beforeOutputs,
                                        const CallBackParam &callParam) -> bool {
      auto diverg_info_map = calibrator_->GetInputDivergInfo();
      if (diverg_info_map->find(callParam.node_name) == diverg_info_map->end()) {
        return true;
      }
      if (PostTrainingQuantizer::CheckFp32TensorVec(callParam.node_name, beforeInputs) != RET_OK) {
        return false;
      }
      if ((*diverg_info_map)[callParam.node_name].size() == 1 &&
          (callParam.node_type == kTypeConcat || callParam.node_type == kTypeAdd)) {
        for (size_t i = 1; i < beforeInputs.size(); i++) {
          auto input_diverg = std::make_unique<DivergInfo>();
          *input_diverg = *((*diverg_info_map)[callParam.node_name][0]);
          (*diverg_info_map)[callParam.node_name].push_back(std::move(input_diverg));
        }
      }
      for (size_t i = 0; i < (*diverg_info_map)[callParam.node_name].size(); i++) {
        auto tensor = beforeInputs[i];
        const auto *tensor_data = static_cast<const float *>(tensor->MutableData());
        size_t elem_count = tensor->ElementsNum();
        vector<float> data(tensor_data, tensor_data + elem_count);
        this->calibrator_->RecordMaxValue(data, (*diverg_info_map)[callParam.node_name][i]);
      }
      return true;
    };
    // func
    KernelCallBack afterCallBack = [&](const std::vector<mindspore::tensor::MSTensor *> &afterInputs,
                                       const std::vector<mindspore::tensor::MSTensor *> &afterOutputs,
                                       const CallBackParam &callParam) -> bool {
      auto diverg_info_map = calibrator_->GetOutputDivergInfo();
      if (diverg_info_map->find(callParam.node_name) == diverg_info_map->end()) {
        return true;
      }
      if (PostTrainingQuantizer::CheckFp32TensorVec(callParam.node_name, afterOutputs) != RET_OK) {
        return false;
      }
      if ((*diverg_info_map)[callParam.node_name].size() == 1 && afterOutputs.size() > 1) {
        for (size_t i = 1; i < afterOutputs.size(); i++) {
          auto output_diverg = std::make_unique<DivergInfo>();
          *output_diverg = *((*diverg_info_map)[callParam.node_name][0]);
          (*diverg_info_map)[callParam.node_name].push_back(std::move(output_diverg));
        }
      }
      size_t output_i = 0;
      for (const auto &tensor : afterOutputs) {
        const auto *tensor_data = static_cast<const float *>(tensor->MutableData());
        size_t elem_count = tensor->ElementsNum();
        vector<float> data(tensor_data, tensor_data + elem_count);
        this->calibrator_->RecordMaxValue(data, (*diverg_info_map)[callParam.node_name][output_i]);
        output_i++;
      }
      return true;
    };
    auto status = fp32_session_->RunGraph(beforeCallBack, afterCallBack);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "run model failed!";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

STATUS PostTrainingQuantizer::Int8Inference() {
  // int8 inference
  vector<mindspore::tensor::MSTensor *> inputs = int8_session_->GetInputs();
  for (auto input_tensor : inputs) {
    // get input tensor
    auto elem_count = input_tensor->ElementsNum();
    vector<float> dummy_data(elem_count);
    std::fill(dummy_data.begin(), dummy_data.end(), 0.1);
    auto ret =
      memcpy_s(input_tensor->MutableData(), input_tensor->Size(), dummy_data.data(), sizeof(float) * dummy_data.size());
    if (ret != EOK) {
      MS_LOG(ERROR) << "memcpy_s error: " << ret;
      return RET_ERROR;
    }
  }

  for (size_t i = 0; i < calibrator_->GetBatchNum(); i++) {
    KernelCallBack beforeCallBack = [this](const std::vector<mindspore::tensor::MSTensor *> &beforeInputs,
                                           const std::vector<mindspore::tensor::MSTensor *> &beforeOutputs,
                                           const CallBackParam &callParam) -> bool {
      if (callParam.node_type == kTypeConv2D || callParam.node_type == kTypeDepthwiseConv2D) {
        vector<float> fp32_op_input;
        while (!OpInputDataHandle(FETCH, callParam.node_name, &fp32_op_input)) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        auto tensor = beforeInputs[0];
        auto lite_tensor = dynamic_cast<mindspore::lite::Tensor *>(tensor);

        if (tensor->data_type() != kNumberTypeInt8) {
          MS_LOG(ERROR) << "unexpected tensor type: " << tensor->data_type();
          return false;
        }

        // do quantization: activation is always per layer quantized
        std::vector<int8_t> quant_datas;
        auto quant_params = lite_tensor->GetQuantParams();
        if (quant_params.size() != 1) {
          MS_LOG(ERROR) << "unexpected quant_params size: " << quant_params.size();
          return false;
        }
        schema::QuantParamT quant_param_t;
        quant_param_t.scale = quant_params[0].scale;
        quant_param_t.zeroPoint = quant_params[0].zeroPoint;
        for (auto float_data : fp32_op_input) {
          auto quant_data = QuantizeData<int8_t>(float_data, quant_param_t, quant_max, quant_min);
          quant_datas.push_back(quant_data);
        }

        if (tensor->Size() != quant_datas.size() * sizeof(int8_t)) {
          MS_LOG(ERROR) << "unexpected tensor size: " << quant_datas.size()
                        << " not the same with: " << quant_datas.size() * sizeof(int8_t);
          return false;
        }

        auto ret =
          memcpy_s(tensor->MutableData(), tensor->Size(), quant_datas.data(), quant_datas.size() * sizeof(int8_t));
        if (ret != EOK) {
          MS_LOG(ERROR) << "memcpy error: " << ret;
          return false;
        }
      }
      return true;
    };
    // func
    KernelCallBack afterCallBack = [this](const std::vector<mindspore::tensor::MSTensor *> &afterInputs,
                                          const std::vector<mindspore::tensor::MSTensor *> &afterOutputs,
                                          const CallBackParam &callParam) -> bool {
      if (callParam.node_type == kTypeConv2D || callParam.node_type == kTypeDepthwiseConv2D) {
        vector<float> fp32_op_output_ch_mean;
        while (!OpOutputChMeanDataHandle(FETCH, callParam.node_name, &fp32_op_output_ch_mean)) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        auto tensor = afterOutputs[0];
        auto lite_tensor = dynamic_cast<mindspore::lite::Tensor *>(tensor);

        if (tensor->data_type() != kNumberTypeInt8) {
          MS_LOG(ERROR) << "unexpected tensor type: " << tensor->data_type();
          return false;
        }

        const int8_t *tensor_data = static_cast<int8_t *>(tensor->MutableData());
        size_t elem_count = tensor->ElementsNum();
        auto shapes = tensor->shape();
        if (shapes.size() != 4) {
          MS_LOG(ERROR) << "unexpected shape size: " << shapes.size();
          return false;
        }
        // suppose the the format is NHWC
        auto channels = shapes[3];
        if (channels == 0) {
          MS_LOG(ERROR) << "unexpected channels: 0";
          return false;
        }
        auto quant_params = lite_tensor->GetQuantParams();
        if (quant_params.size() != 1) {
          MS_LOG(ERROR) << "unexpected activatation quant_params size: " << quant_params.size();
          return false;
        }
        auto scale = quant_params[0].scale;
        auto zp = quant_params[0].zeroPoint;

        std::vector<float> dequant_op_output_ch_mean(channels);
        auto one_filter_size = elem_count / channels;
        for (int i = 0; i < channels; i++) {
          float sum = 0;
          for (size_t j = 0; j < one_filter_size; j++) {
            auto index = j * channels + i;
            if (index >= elem_count) {
              MS_LOG(ERROR) << "over flow!";
              return RET_ERROR;
            }
            // deuqant activation
            auto float_data = scale * (tensor_data[index] - zp);
            sum += float_data;
          }
          sum = sum / one_filter_size;
          dequant_op_output_ch_mean[i] = sum;
        }
        std::transform(fp32_op_output_ch_mean.begin(), fp32_op_output_ch_mean.end(), dequant_op_output_ch_mean.begin(),
                       dequant_op_output_ch_mean.begin(), std::minus<>());

        if (op_bias_diff_map.find(callParam.node_name) != op_bias_diff_map.end()) {
          auto &bias_diff = op_bias_diff_map[callParam.node_name];
          std::transform(bias_diff.begin(), bias_diff.end(), dequant_op_output_ch_mean.begin(), bias_diff.begin(),
                         std::plus<>());
        } else {
          op_bias_diff_map[callParam.node_name] = dequant_op_output_ch_mean;
        }
      }
      return true;
    };
    auto ret = int8_session_->RunGraph(beforeCallBack, afterCallBack);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "run model failed!";
      return RET_ERROR;
    }
  }  // end for images
  return RET_OK;
}

STATUS PostTrainingQuantizer::BiasCorrection(const FuncGraphPtr &func_graph) {
  auto ret = RET_OK;
  std::future<STATUS> int8_inference = std::async(std::launch::async, &PostTrainingQuantizer::Int8Inference, this);
  // get input tensor
  vector<mindspore::tensor::MSTensor *> inputs = fp32_session_->GetInputs();
  if (inputs.size() != 1) {
    MS_LOG(ERROR) << "model's input tensor size: " << inputs.size();
    return RET_ERROR;
  }
  // fp32 inference
  for (size_t i = 0; i < calibrator_->GetBatchNum(); i++) {
    for (size_t input_index = 0; input_index < inputs.size(); input_index++) {
      STATUS status = calibrator_->GenerateInputData(input_index, i, inputs[input_index]);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "generate input data from images failed!";
        return RET_ERROR;
      }
    }
    KernelCallBack beforeCallBack = [this](const std::vector<mindspore::tensor::MSTensor *> &beforeInputs,
                                           const std::vector<mindspore::tensor::MSTensor *> &beforeOutputs,
                                           const CallBackParam &callParam) -> bool {
      if (callParam.node_type == kTypeConv2D || callParam.node_type == kTypeDepthwiseConv2D) {
        if (PostTrainingQuantizer::CheckFp32TensorVec(callParam.node_name, beforeInputs) != RET_OK) {
          return false;
        }
        auto tensor = beforeInputs[0];
        size_t elem_count = tensor->ElementsNum();
        std::vector<float> fp32_op_input(elem_count);
        auto ret =
          memcpy_s(fp32_op_input.data(), fp32_op_input.size() * sizeof(float), tensor->MutableData(), tensor->Size());
        if (ret != EOK) {
          MS_LOG(ERROR) << "memcpy error: " << ret;
          return false;
        }
        while (!OpInputDataHandle(STORE, callParam.node_name, &fp32_op_input)) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
      }
      return true;
    };
    // func
    KernelCallBack afterCallBack = [this](const std::vector<mindspore::tensor::MSTensor *> &afterInputs,
                                          const std::vector<mindspore::tensor::MSTensor *> &afterOutputs,
                                          const CallBackParam &callParam) -> bool {
      if (callParam.node_type == kTypeConv2D || callParam.node_type == kTypeDepthwiseConv2D) {
        if (PostTrainingQuantizer::CheckFp32TensorVec(callParam.node_name, afterOutputs) != RET_OK) {
          return false;
        }
        auto tensor = afterOutputs[0];
        const auto *tensor_data = static_cast<const float *>(tensor->MutableData());
        size_t elem_count = tensor->ElementsNum();
        auto shapes = tensor->shape();
        if (shapes.size() != 4) {
          MS_LOG(ERROR) << "unexpected shape size: " << shapes.size();
          return false;
        }
        // suppose the activation format: NHWC
        auto channels = shapes[3];
        if (channels == 0) {
          MS_LOG(ERROR) << "unexpected channels: 0";
          return false;
        }
        std::vector<float> fp32_op_output_ch_mean(channels);
        auto one_filter_size = elem_count / channels;
        for (int i = 0; i < channels; i++) {
          float sum = 0;
          for (size_t j = 0; j < one_filter_size; j++) {
            auto index = j * channels + i;
            if (index >= elem_count) {
              MS_LOG(ERROR) << "over flow!";
              return RET_ERROR;
            }
            sum += tensor_data[index];
          }
          sum = sum / one_filter_size;
          fp32_op_output_ch_mean[i] = sum;
        }
        while (!OpOutputChMeanDataHandle(STORE, callParam.node_name, &fp32_op_output_ch_mean)) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
      }

      return true;
    };
    auto status = fp32_session_->RunGraph(beforeCallBack, afterCallBack);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "run model failed!";
      return RET_ERROR;
    }
  }  // end for images

  ret = int8_inference.get();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "int8 inference failed!";
    return RET_ERROR;
  }
  for (auto &key_value : op_bias_diff_map) {
    std::for_each(key_value.second.begin(), key_value.second.end(),
                  [this](float &data) { data = data / calibrator_->GetBatchNum(); });
  }
  auto cnodes = func_graph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    auto op_name = cnode->fullname_with_scope();
    if (op_bias_diff_map.find(op_name) != op_bias_diff_map.end()) {
      const auto &bias_diff = op_bias_diff_map[op_name];
      auto primitive_c = GetValueNode<std::shared_ptr<PrimitiveC>>(cnode->input(0));
      if (primitive_c == nullptr) {
        MS_LOG(ERROR) << "primitive_c is nullptr";
        continue;
      }
      auto input_quant_params = primitive_c->GetInputQuantParams();

      if (input_quant_params.size() == 3) {
        // compensate the existed
        auto bias_quant_params = input_quant_params[2];
        auto bias = cnode->input(3);
        auto bias_parameter_ptr = std::dynamic_pointer_cast<Parameter>(bias);
        auto bias_default_param = bias_parameter_ptr->default_param();
        auto bias_param = std::dynamic_pointer_cast<ParamValueLite>(bias_default_param);
        int *bias_datas = static_cast<int *>(bias_param->tensor_addr());

        if (static_cast<size_t>(bias_param->tensor_shape_size()) != bias_diff.size()) {
          MS_LOG(ERROR) << "unexpected bias data count: " << bias_param->tensor_shape_size()
                        << " not the same as bias_diff: " << bias_diff.size();
          continue;
        }
        if (bias_quant_params.size() != bias_diff.size()) {
          MS_LOG(ERROR) << "unexpected bias quant params size: " << bias_quant_params.size()
                        << " not the same as bias_diff: " << bias_diff.size();
        }

        for (int i = 0; i < bias_param->tensor_shape_size(); i++) {
          auto scale = bias_quant_params[i].scale;
          double after_correct = std::round(bias_diff[i] / scale) + bias_datas[i];
          const constexpr int32_t corrected_bias_abs_limit = 0.6 * INT32_MAX;
          if (after_correct > corrected_bias_abs_limit) {
            MS_LOG(WARNING) << op_name << " ch: " << i << " bias after_corrected too large: " << after_correct
                            << " origin value: " << bias_datas[i] << " bias_diff: " << bias_diff[i]
                            << " scale: " << scale;
            bias_datas[i] = static_cast<int>(corrected_bias_abs_limit);
          } else if (after_correct < -corrected_bias_abs_limit) {
            MS_LOG(WARNING) << op_name << " ch: " << i << " bias after_corrected too small: " << after_correct
                            << " origin value: " << bias_datas[i] << " bias_diff: " << bias_diff[i]
                            << " scale: " << scale;
            bias_datas[i] = static_cast<int>(-corrected_bias_abs_limit);
          } else {
            auto diff = static_cast<int>(std::round(bias_diff[i] / scale));
            bias_datas[i] += diff;
          }
        }
      } else if (input_quant_params.size() == 2) {
        MS_LOG(INFO) << op_name << " add bias input";
        // need to add bias input
        auto parameter = func_graph->add_parameter();
        ShapeVector shape;
        shape.push_back(bias_diff.size());
        auto type_ptr = TypeIdToType(kNumberTypeFloat32);
        auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, shape);
        parameter->set_abstract(abstract_tensor);
        parameter->set_name("added_" + op_name + "_bias");

        ParamValueLitePtr param_value = std::make_shared<ParamValueLite>();
        MS_ASSERT(param_value != nullptr);
        std::vector<int32_t> shape_vector;
        (void)std::transform(shape.begin(), shape.end(), std::back_inserter(shape_vector),
                             [](const int64_t &value) { return static_cast<int32_t>(value); });
        param_value->set_tensor_shape(shape_vector);
        param_value->set_tensor_type(kNumberTypeFloat32);

        auto size = sizeof(float) * bias_diff.size();
        char *tensor_data = new (std::nothrow) char[size];
        if (tensor_data == nullptr) {
          MS_LOG(ERROR) << "new char[] failed";
          return RET_MEMORY_FAILED;
        }
        ret = ::memcpy_s(tensor_data, size * sizeof(char), bias_diff.data(), size * sizeof(char));
        if (ret != EOK) {
          MS_LOG(ERROR) << "memcpy_s error: " << ret;
          free(tensor_data);
          tensor_data = nullptr;
          return false;
        }
        param_value->set_tensor_addr(tensor_data);
        param_value->set_tensor_size(size);
        parameter->set_default_param(param_value);
        cnode->add_input(parameter);
        DoBiasQuant(parameter, primitive_c);

        auto op_type = (schema::PrimitiveType)primitive_c->Type();
        if (op_type == schema::PrimitiveType_Conv2D) {
          auto conv2d = primitive_c->GetPrimitiveT()->value.AsConv2D();
          if (conv2d == nullptr) {
            MS_LOG(ERROR) << "conv2d is null";
            free(tensor_data);
            tensor_data = nullptr;
            return RET_ERROR;
          }
          conv2d->hasBias = true;
        } else if (op_type == schema::PrimitiveType_DepthwiseConv2D) {
          auto depthwise_conv2d = primitive_c->GetPrimitiveT()->value.AsDepthwiseConv2D();
          if (depthwise_conv2d == nullptr) {
            MS_LOG(ERROR) << "conv2d is null";
            free(tensor_data);
            tensor_data = nullptr;
            return RET_ERROR;
          }
          depthwise_conv2d->hasBias = true;
        }
        free(tensor_data);
        tensor_data = nullptr;
      } else {
        MS_LOG(ERROR) << "unexpected input_quant_params size: " << input_quant_params.size();
        continue;
      }
    }  // end fine op_name
  }

  return ret;
}

STATUS PostTrainingQuantizer::CollectDataFrequency() {
  // get input tensor
  vector<mindspore::tensor::MSTensor *> inputs = fp32_session_->GetInputs();
  if (inputs.size() != calibrator_->GetInputNum()) {
    MS_LOG(ERROR) << "model's input tensor cnt: " << inputs.size() << " != " << calibrator_->GetInputNum();
    return RET_ERROR;
  }

  for (size_t i = 0; i < calibrator_->GetBatchNum(); i++) {
    // set multi-input data
    for (size_t input_index = 0; input_index < inputs.size(); input_index++) {
      STATUS status = calibrator_->GenerateInputData(input_index, i, inputs[input_index]);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "generate input data from images failed!";
        return RET_ERROR;
      }
    }

    KernelCallBack beforeCallBack = [&](const std::vector<mindspore::tensor::MSTensor *> &beforeInputs,
                                        const std::vector<mindspore::tensor::MSTensor *> &beforeOutputs,
                                        const CallBackParam &callParam) {
      auto diverg_info_map = calibrator_->GetInputDivergInfo();
      if (diverg_info_map->find(callParam.node_name) == diverg_info_map->end()) {
        return true;
      }
      if (PostTrainingQuantizer::CheckFp32TensorVec(callParam.node_name, beforeInputs) != RET_OK) {
        return false;
      }
      for (size_t i = 0; i < (*diverg_info_map)[callParam.node_name].size(); i++) {
        auto tensor = beforeInputs[i];
        const auto *tensor_data = static_cast<const float *>(tensor->MutableData());
        size_t elem_count = tensor->ElementsNum();
        vector<float> data(tensor_data, tensor_data + elem_count);
        this->calibrator_->UpdateDataFrequency(data, (*diverg_info_map)[callParam.node_name][i]);
      }
      return true;
    };

    KernelCallBack afterCallBack = [&](const std::vector<mindspore::tensor::MSTensor *> &after_inputs,
                                       const std::vector<mindspore::tensor::MSTensor *> &after_outputs,
                                       const CallBackParam &call_param) {
      auto diverg_info_map = calibrator_->GetOutputDivergInfo();
      if (diverg_info_map->find(call_param.node_name) == diverg_info_map->end()) {
        return true;
      }
      if (PostTrainingQuantizer::CheckFp32TensorVec(call_param.node_name, after_outputs) != RET_OK) {
        return false;
      }
      int output_i = 0;
      for (const auto &tensor : after_outputs) {
        const auto *tensor_data = static_cast<const float *>(tensor->MutableData());
        size_t elem_count = tensor->ElementsNum();
        vector<float> data(tensor_data, tensor_data + elem_count);
        this->calibrator_->UpdateDataFrequency(data, (*diverg_info_map)[call_param.node_name][output_i]);
        output_i++;
      }
      return true;
    };
    auto status = fp32_session_->RunGraph(beforeCallBack, afterCallBack);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "run model failed!";
      return RET_ERROR;
    }
  }

  return RET_OK;
}

STATUS PostTrainingQuantizer::ComputeThreshold() { return this->calibrator_->ComputeThreshold(); }

STATUS PostTrainingQuantizer::DoQuantize(FuncGraphPtr func_graph) {
  MS_LOG(INFO) << "start to parse config file";
  STATUS status = PreProcess();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "do pre process failed!";
    return status;
  }
  // anf -- fb
  auto meta_graph = Export(func_graph, true, true);
  if (meta_graph == nullptr) {
    MS_LOG(ERROR) << "Export to meta_graph return nullptr";
    return RET_ERROR;
  }

  // transform
  GraphDefTransform transform;
  transform.SetGraphDef(meta_graph);
  flags.quantType = schema::QuantType_QUANT_NONE;
  status = transform.Transform(flags);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "FBTransform model failed " << status;
    return RET_ERROR;
  }
  MS_LOG(INFO) << "start create session";
  flatbuffers::FlatBufferBuilder builder(1024);
  auto offset = schema::MetaGraph::Pack(builder, meta_graph);
  builder.Finish(offset);
  size_t size = builder.GetSize();
  auto *content = reinterpret_cast<const char *>(builder.GetBufferPointer());
  if (content == nullptr) {
    MS_LOG(ERROR) << "GetBufferPointer nullptr";
    return RET_ERROR;
  }
  auto model = lite::Model::Import(content, size);

  Context ctx;
  ctx.thread_num_ = calibrator_->GetThreadNum();

  fp32_session_ = dynamic_cast<mindspore::lite::LiteSession *>(session::LiteSession::CreateSession(&ctx));
  if (fp32_session_ == nullptr) {
    MS_LOG(ERROR) << "create session failed!";
    return RET_ERROR;
  }

  auto ret = fp32_session_->CompileGraph(model);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "compile graph error";
    return RET_ERROR;
  }

  MS_LOG(INFO) << "start to update divergence's max value";
  status = DoInference();
  if (status != RET_OK) {
    return status;
  }
  MS_LOG(INFO) << "start to update divergence's interval";
  status = UpdateDivergInverval();
  if (status != RET_OK) {
    return status;
  }
  MS_LOG(INFO) << "start to collect data's distribution";
  status = CollectDataFrequency();
  if (status != RET_OK) {
    return status;
  }
  MS_LOG(INFO) << "compute the best threshold";
  status = ComputeThreshold();
  if (status != RET_OK) {
    return status;
  }
  MS_LOG(INFO) << "start to generate quant param and quantize tensor's data";
  status = QuantNode();
  if (status != RET_OK) {
    return status;
  }

  // add quant_cast
  quant::QuantCast quant_cast;
  quant_cast.SetInputDataDType(kNumberTypeFloat32);
  status = quant_cast.Run(func_graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "add QuantCast error";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
    return RET_ERROR;
  }

  if (calibrator_->GetBiasCorrection()) {
    // init in8 session
    // anf -- fb
    auto int8_meta_graph = Export(func_graph, true, true);
    if (int8_meta_graph == nullptr) {
      MS_LOG(ERROR) << "Export to int8_meta_graph return nullptr";
      return RET_ERROR;
    }

    // transform
    GraphDefTransform fb_transform;
    fb_transform.SetGraphDef(int8_meta_graph);
    flags.quantType = schema::QuantType_PostTraining;
    status = fb_transform.Transform(flags);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "FBTransform model failed " << status;
      return RET_ERROR;
    }
    MS_LOG(INFO) << "start create quantized session";
    flatbuffers::FlatBufferBuilder int8_builder(1024);
    auto int8_offset = schema::MetaGraph::Pack(int8_builder, int8_meta_graph);
    int8_builder.Finish(int8_offset);
    size = int8_builder.GetSize();
    auto *int8_content = reinterpret_cast<const char *>(int8_builder.GetBufferPointer());
    if (int8_content == nullptr) {
      MS_LOG(ERROR) << "GetBufferPointer nullptr";
      return RET_ERROR;
    }
    auto int8_model = lite::Model::Import(int8_content, size);

    Context int8_ctx;
    int8_ctx.thread_num_ = calibrator_->GetThreadNum();
    int8_ctx.device_list_[0].device_info_.cpu_device_info_.cpu_bind_mode_ = HIGHER_CPU;

    int8_session_ = dynamic_cast<mindspore::lite::LiteSession *>(session::LiteSession::CreateSession(&int8_ctx));
    if (int8_session_ == nullptr) {
      MS_LOG(ERROR) << "create session failed!";
      return RET_ERROR;
    }
    ret = int8_session_->CompileGraph(int8_model);
    if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "compile graph error";
      return RET_ERROR;
    }

    MS_LOG(INFO) << "do bias correction";
    status = BiasCorrection(func_graph);
    if (status != RET_OK) {
      MS_LOG(WARNING) << "BiasCorrection failed.";
    }
  }

  return RET_OK;
}

bool PostTrainingQuantizer::OpInputDataHandle(OperationType type, const string &op_name, std::vector<float> *data) {
  std::lock_guard<std::mutex> lg(mutex_op_input);
  if (type == STORE) {
    if (fp32_op_input_map.find(op_name) != fp32_op_input_map.end()) {
      // the data has not been fetched by int8 model
      return false;
    }
    fp32_op_input_map[op_name] = *data;
    return true;
  } else if (type == FETCH) {
    if (fp32_op_input_map.find(op_name) == fp32_op_input_map.end()) {
      // the data not generated by fp32 model yet
      return false;
    }
    *data = fp32_op_input_map[op_name];
    fp32_op_input_map.erase(op_name);
    return true;
  } else {
    MS_LOG(ERROR) << "unexpected type: " << type;
  }
  return false;
}

bool PostTrainingQuantizer::OpOutputChMeanDataHandle(OperationType type, const string &op_name,
                                                     std::vector<float> *data) {
  std::lock_guard<std::mutex> lg(mutex_op_output);
  if (type == STORE) {
    if (fp32_op_output_ch_mean_map.find(op_name) != fp32_op_output_ch_mean_map.end()) {
      // the data has not been fetched by int8 model
      return false;
    }
    fp32_op_output_ch_mean_map[op_name] = *data;
    return true;
  } else if (type == FETCH) {
    if (fp32_op_output_ch_mean_map.find(op_name) == fp32_op_output_ch_mean_map.end()) {
      // the data not generated by fp32 model yet
      return false;
    }
    *data = fp32_op_output_ch_mean_map[op_name];
    fp32_op_output_ch_mean_map.erase(op_name);
    return true;
  } else {
    MS_LOG(ERROR) << "unexpected type: " << type;
  }
  return false;
}

}  // namespace quant
}  // namespace lite
}  // namespace mindspore

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
#include "minddata/dataset/core/config_manager.h"

#include <fstream>
#include <iostream>
#include <string>
#include <utility>

#ifndef ENABLE_ANDROID
#include "utils/log_adapter.h"
#else
#include "mindspore/lite/src/common/log_adapter.h"
#endif
#include "minddata/dataset/util/system_pool.h"

namespace mindspore {
namespace dataset {
ConfigManager::ConfigManager()
    : rows_per_buffer_(kCfgRowsPerBuffer),
      num_parallel_workers_(kCfgParallelWorkers),
      worker_connector_size_(kCfgWorkerConnectorSize),
      op_connector_size_(kCfgOpConnectorSize),
      seed_(kCfgDefaultSeed),
      monitor_sampling_interval_(kCfgMonitorSamplingInterval),
      callback_timout_(kCfgCallbackTimeout),
      cache_host_(kCfgDefaultCacheHost),
      cache_port_(kCfgDefaultCachePort),
      num_connections_(kDftNumConnections),
      prefetch_size_(kDftPrefetchSize) {
  auto env_cache_host = std::getenv("MS_CACHE_HOST");
  auto env_cache_port = std::getenv("MS_CACHE_PORT");
  if (env_cache_host != nullptr) {
    cache_host_ = env_cache_host;
  }
  if (env_cache_port != nullptr) {
    char *end = nullptr;
    cache_port_ = strtol(env_cache_port, &end, 10);
    if (*end != '\0') {
      MS_LOG(WARNING) << "\nCache port from env variable MS_CACHE_PORT is invalid, back to use default "
                      << kCfgDefaultCachePort << std::endl;
      cache_port_ = kCfgDefaultCachePort;
    }
  }
}

// A print method typically used for debugging
void ConfigManager::Print(std::ostream &out) const {
  // Don't show the test/internal ones.  Only display the main ones here.
  // fyi, boolalpha tells the output stream to write "true" and "false" for bools
  out << "\nClient config settings :"
      << "\nDataCache Rows per buffer    : " << rows_per_buffer_
      << "\nParallelOp workers           : " << num_parallel_workers_
      << "\nParallelOp worker connector size    : " << worker_connector_size_
      << "\nSize of each Connector : " << op_connector_size_ << std::endl;
}

// Private helper function that taks a nlohmann json format and populates the settings
Status ConfigManager::FromJson(const nlohmann::json &j) {
  set_rows_per_buffer(j.value("rowsPerBuffer", rows_per_buffer_));
  set_num_parallel_workers(j.value("numParallelWorkers", num_parallel_workers_));
  set_worker_connector_size(j.value("workerConnectorSize", worker_connector_size_));
  set_op_connector_size(j.value("opConnectorSize", op_connector_size_));
  set_seed(j.value("seed", seed_));
  set_monitor_sampling_interval(j.value("monitorSamplingInterval", monitor_sampling_interval_));
  set_cache_host(j.value("cacheHost", cache_host_));
  set_cache_port(j.value("cachePort", cache_port_));
  set_num_connections(j.value("numConnections", num_connections_));
  set_prefetch_size(j.value("prefetchSize", prefetch_size_));
  return Status::OK();
}

// Loads a json file with the default settings and populates all the settings
Status ConfigManager::LoadFile(const std::string &settingsFile) {
  Status rc;
  if (!Path(settingsFile).Exists()) {
    RETURN_STATUS_UNEXPECTED("File is not found.");
  }
  // Some settings are mandatory, others are not (with default).  If a setting
  // is optional it will set a default value if the config is missing from the file.
  try {
    std::ifstream in(settingsFile);
    nlohmann::json js;
    in >> js;
    rc = FromJson(js);
  } catch (const nlohmann::json::type_error &e) {
    std::ostringstream ss;
    ss << "Client file failed to load:\n" << e.what();
    std::string err_msg = ss.str();
    RETURN_STATUS_UNEXPECTED(err_msg);
  } catch (const std::exception &err) {
    RETURN_STATUS_UNEXPECTED("Client file failed to load.");
  }
  return rc;
}

// Setter function
void ConfigManager::set_rows_per_buffer(int32_t rows_per_buffer) { rows_per_buffer_ = rows_per_buffer; }

// Setter function
void ConfigManager::set_num_parallel_workers(int32_t num_parallel_workers) {
  num_parallel_workers_ = num_parallel_workers;
}

// Setter function
void ConfigManager::set_worker_connector_size(int32_t connector_size) { worker_connector_size_ = connector_size; }

// Setter function
void ConfigManager::set_op_connector_size(int32_t connector_size) { op_connector_size_ = connector_size; }

uint32_t ConfigManager::seed() const { return seed_; }

void ConfigManager::set_seed(uint32_t seed) { seed_ = seed; }

void ConfigManager::set_monitor_sampling_interval(uint32_t interval) { monitor_sampling_interval_ = interval; }

void ConfigManager::set_callback_timeout(uint32_t timeout) { callback_timout_ = timeout; }

void ConfigManager::set_cache_host(std::string cache_host) { cache_host_ = std::move(cache_host); }

void ConfigManager::set_cache_port(int32_t cache_port) { cache_port_ = cache_port; }

void ConfigManager::set_num_connections(int32_t num_connections) { num_connections_ = num_connections; }

void ConfigManager::set_prefetch_size(int32_t prefetch_size) { prefetch_size_ = prefetch_size; }
}  // namespace dataset
}  // namespace mindspore

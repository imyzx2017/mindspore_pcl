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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_CONFIG_MANAGER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_CONFIG_MANAGER_H_

#include <ostream>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>

#include "minddata/dataset/core/constants.h"
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/status.h"

// Config settings for the client-side
// example config file:
// {
//    "rowsPerBuffer": 3
// }
//

namespace mindspore {
namespace dataset {
// The ConfigManager is a class for managing default values.  When a user is constructing any objects
// in the framework, often they may choose to omit some settings instead of overriding them.
// This class manages some of the default values, for cases when the user does not manually specify
// those values.
class ConfigManager {
 public:
  ConfigManager();

  // destructor
  ~ConfigManager() = default;

  // A print method typically used for debugging
  // @param out - The output stream to write output to
  void Print(std::ostream &out) const;

  // << Stream output operator overload
  // @notes This allows you to write the debug print info using stream operators
  // @param out - reference to the output stream being overloaded
  // @param cS - reference to the ConfigManager to display
  // @return - the output stream must be returned
  friend std::ostream &operator<<(std::ostream &out, const ConfigManager &cS) {
    cS.Print(out);
    return out;
  }

  // Another debug print helper.  Converts the print info to a string for you.
  // @return The string version of the debug print
  std::string ToString() {
    std::stringstream ss;
    ss << *this;
    return ss.str();
  }

  // Loads a json file with the default settings and populates all the settings
  // @param settingsFile - A json file with a set of default settings
  // @return Status error code
  Status LoadFile(const std::string &settingsFile);

  // getter function
  // @return The rows per buffer setting
  int32_t rows_per_buffer() const { return rows_per_buffer_; }

  // getter function
  // @return The number of workers setting
  int32_t num_parallel_workers() const { return num_parallel_workers_; }

  // getter function
  // @return The queue size of the operator's output connector
  int32_t op_connector_size() const { return op_connector_size_; }

  // getter function
  // @return The internal worker-to-master connector queue size
  int32_t worker_connector_size() const { return worker_connector_size_; }

  // getter function
  // @return The hostname of cache server
  std::string cache_host() const { return cache_host_; }

  // getter function
  // @return The port of cache server
  int32_t cache_port() const { return cache_port_; }

  /// getter function
  /// \return Number of tcp/ip connection
  int32_t num_connections() const { return num_connections_; }

  /// getter function
  /// \return Prefetch size
  int32_t prefetch_size() const { return prefetch_size_; }

  // setter function
  // @param rows_per_buffer - The setting to apply to the config
  void set_rows_per_buffer(int32_t rows_per_buffer);

  // setter function
  // @param num_parallel_workers - The setting to apply to the config
  void set_num_parallel_workers(int32_t num_parallel_workers);

  // setter function
  // @param connector_size - The setting to apply to the config
  void set_worker_connector_size(int32_t connector_size);

  // setter function
  // @param connector_size - The setting to apply to the config
  void set_op_connector_size(int32_t connector_size);

  // setter function
  // @param cache_host - The hostname of cache server
  void set_cache_host(std::string cache_host);

  // setter function
  // @param cache_port - The port of cache server
  void set_cache_port(int32_t cache_port);

  /// setter function
  /// \param num_connections
  void set_num_connections(int32_t num_connections);

  /// setter function
  /// \param prefetch_size
  void set_prefetch_size(int32_t prefetch_size);

  uint32_t seed() const;

  // setter function
  // @param seed - The default seed to use
  void set_seed(uint32_t seed);

  // setter function
  // @param interval - The setting to apply to the config
  void set_monitor_sampling_interval(uint32_t interval);

  // getter function
  // @return The interval of monitor sampling
  int32_t monitor_sampling_interval() const { return monitor_sampling_interval_; }

  // setter function
  // @param timeout - The setting to apply to the config
  void set_callback_timeout(uint32_t timeout);

  // getter function
  // @return The timeout DSWaitedCallback would wait for before raising an error
  int32_t callback_timeout() const { return callback_timout_; }

 private:
  int32_t rows_per_buffer_;
  int32_t num_parallel_workers_;
  int32_t worker_connector_size_;
  int32_t op_connector_size_;
  uint32_t seed_;
  uint32_t monitor_sampling_interval_;
  uint32_t callback_timout_;
  std::string cache_host_;
  int32_t cache_port_;
  int32_t num_connections_;
  int32_t prefetch_size_;

  // Private helper function that takes a nlohmann json format and populates the settings
  // @param j - The json nlohmann json info
  Status FromJson(const nlohmann::json &j);
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_CONFIG_MANAGER_H_

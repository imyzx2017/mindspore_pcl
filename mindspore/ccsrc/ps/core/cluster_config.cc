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

#include "ps/core/cluster_config.h"

#include <string>

namespace mindspore {
namespace ps {
namespace core {

uint32_t ClusterConfig::worker_num_ = 0;
uint32_t ClusterConfig::server_num_ = 0;
uint32_t ClusterConfig::heartbeat_interval_ = kHeartbeatInterval;
std::unique_ptr<std::string> ClusterConfig::scheduler_host_ = nullptr;
uint16_t ClusterConfig::scheduler_port_ = 0;

void ClusterConfig::Init(const uint32_t &worker_num, const uint32_t &server_num,
                         std::unique_ptr<std::string> scheduler_host, const uint16_t &scheduler_port) {
  worker_num_ = worker_num;
  server_num_ = server_num;
  if (!CommUtil::CheckIp(*scheduler_host.get())) {
    MS_LOG(EXCEPTION) << "The scheduler_host:" << *scheduler_host.get() << " is illegal!";
  }
  scheduler_host_ = std::move(scheduler_host);
  scheduler_port_ = scheduler_port;
}

uint32_t ClusterConfig::worker_num() { return worker_num_; }

uint32_t ClusterConfig::server_num() { return server_num_; }

uint32_t ClusterConfig::heartbeat_interval() { return heartbeat_interval_; }

void ClusterConfig::set_heartbeat_interval(const uint32_t &heartbeat_interval) {
  heartbeat_interval_ = heartbeat_interval;
}

std::string ClusterConfig::scheduler_host() { return *scheduler_host_.get(); }

uint16_t ClusterConfig::scheduler_port() { return scheduler_port_; }

}  // namespace core
}  // namespace ps
}  // namespace mindspore

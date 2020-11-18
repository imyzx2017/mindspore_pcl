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

#ifndef MINDSPORE_CCSRC_PS_CONTEXT_H_
#define MINDSPORE_CCSRC_PS_CONTEXT_H_

#include <string>
#include <memory>

namespace mindspore {
namespace ps {
constexpr char kEnvRole[] = "MS_ROLE";
constexpr char kEnvRoleOfPServer[] = "MS_PSERVER";
constexpr char kEnvRoleOfWorker[] = "MS_WORKER";
constexpr char kEnvRoleOfScheduler[] = "MS_SCHED";
constexpr char kEnvRoleOfNotPS[] = "MS_NOT_PS";

class PSContext {
 public:
  ~PSContext() = default;
  PSContext(PSContext const &) = delete;
  PSContext &operator=(const PSContext &) = delete;
  static std::shared_ptr<PSContext> instance();

  void SetPSEnable(bool enabled);
  bool is_ps_enabled() const;
  void Reset();
  std::string ms_role() const;
  bool is_role_worker() const;
  bool is_role_pserver() const;
  bool is_role_sched() const;
  void SetPSRankId(int rank_id);
  int ps_rank_id() const;

 private:
  PSContext() : ps_enabled_(false), is_worker_(false), is_pserver_(false), is_sched_(false), rank_id_(-1) {}
  bool ps_enabled_;
  bool is_worker_;
  bool is_pserver_;
  bool is_sched_;
  int rank_id_;
};
}  // namespace ps
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PS_CONTEXT_H_

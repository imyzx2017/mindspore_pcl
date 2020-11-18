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

#include "common/thread_pool.h"
#include <algorithm>
#include "utils/log_adapter.h"
#include "utils/convert_utils_base.h"

namespace mindspore {
#ifdef ENABLE_D
const int kDeviceNum = 8;
#endif

bool Queue::Enqueue(Task *task) {
  const int tail_index = tail_.load(std::memory_order_relaxed);
  // queue full
  auto next = (tail_index + 1) % 2;
  if (next == head_.load(std::memory_order_acquire)) {
    return false;
  }
  buffer_[tail_index] = task;
  tail_.store(next, std::memory_order_release);
  ++task_size_;
  return true;
}

bool Queue::Dequeue(Task **out) {
  if (task_size_ == 0) {
    return false;
  }
  // queue empty
  const int head_index = head_.load(std::memory_order_relaxed);
  if (head_index == tail_.load(std::memory_order_acquire)) {
    return false;
  }
  *out = buffer_[head_index];
  head_.store((head_index + 1) % 2, std::memory_order_release);
  return true;
}

ThreadPool::ThreadPool() {
#ifdef ENABLE_D
  auto cpu_core_num = std::thread::hardware_concurrency();
  max_thread_num_ = cpu_core_num / kDeviceNum;
#endif
  SetThreadPool(core_thread_num_);
}

bool ThreadPool::SetThreadPool(int config_thread_num) {
  std::lock_guard<std::mutex> Lock(pool_mtx_);
  if (config_thread_num > max_thread_num_) {
    MS_LOG(EXCEPTION) << "Expected thread num is greater than the max thread num, expected thread num="
                      << config_thread_num << ", allowed max thread num=" << max_thread_num_;
  }
  if (config_thread_num > cur_thread_nums_) {
    AddNewThread(config_thread_num - cur_thread_nums_);
  }
  MS_LOG(DEBUG) << "cur_thread_nums_=" << cur_thread_nums_ << ", cur_thread_run_nums_=" << cur_thread_run_nums_;
  return true;
}

void ThreadPool::AddNewThread(int add_num) {
  for (int i = cur_thread_nums_, j = 0; j < add_num; ++i, ++j) {
    auto active = new std::atomic_bool{true};
    auto queue = std::make_shared<Queue>();
    std::thread thread([this, i, active, queue]() {
      Task *task = nullptr;
      while (!exit_run_) {
        while (*active) {
          if (queue->Dequeue(&task)) {
            auto ret = (*task)();
            if (ret != SUCCESS) {
              error_info_.emplace_back(std::make_pair(i, std::make_pair(false, ret)));
            }
            queue->task_size_--;
          }
          std::this_thread::yield();
        }
        std::unique_lock<std::mutex> queue_lock(thread_mtx_);
        queue_ready_.wait(queue_lock, [active, this] { return exit_run_ || *active; });
      }
    });
    thread_list_.emplace_back(std::move(thread));
    activate_list_.emplace_back(active);
    queue_list_.emplace_back(queue);
  }
  cur_thread_nums_ += add_num;
  cur_thread_run_nums_ += add_num;
  MS_LOG(INFO) << "add " << add_num << " thread";
}

void ThreadPool::AddRunThread(int num) {
  MS_LOG(DEBUG) << "num=" << num << ", cur_thread_run_nums_=" << cur_thread_run_nums_;
  int active_nums = num - cur_thread_run_nums_;
  if (active_nums <= 0 || static_cast<int>(activate_list_.size()) < active_nums) {
    return;
  }
  for (int i = cur_thread_run_nums_ - 1, j = 0; j < active_nums; ++i, ++j) {
    *activate_list_[i] = true;
  }
  std::lock_guard<std::mutex> queueLock(thread_mtx_);
  queue_ready_.notify_all();
  cur_thread_run_nums_ = num;
}

void ThreadPool::SubRunThread(int num) {
  MS_LOG(DEBUG) << "sub num=" << num << ", cur_thread_run_nums_=" << cur_thread_run_nums_;
  int deactive_nums = cur_thread_run_nums_ - num;
  if (deactive_nums <= 0) {
    return;
  }
  for (int i = num, j = 0; j < deactive_nums; ++i, ++j) {
    *activate_list_[i] = false;
  }
  cur_thread_run_nums_ = num;
}

bool ThreadPool::LaunchMultipleTask(const std::vector<Task> &tasks) {
  int thread_num = tasks.size();
  if (thread_num > max_thread_num_) {
    thread_num = max_thread_num_;
  }
  if (!SetThreadPool(thread_num)) {
    return false;
  }
  error_info_.clear();
  bool succ_flag;
  for (int task_id = 0, queue_index = 0; task_id < SizeToInt(tasks.size()); ++task_id) {
    do {
      succ_flag = true;
      if (!queue_list_[queue_index]->Enqueue(const_cast<Task *>(&tasks[task_id]))) {
        std::this_thread::yield();
        succ_flag = false;
      }
    } while (!succ_flag);
    queue_index++;
    if (queue_index >= cur_thread_run_nums_) {
      queue_index = queue_index - cur_thread_run_nums_;
    }
  }
  succ_flag = false;
  while (!succ_flag) {
    std::this_thread::yield();
    succ_flag = true;
    for (int i = 0; i < cur_thread_run_nums_; ++i) {
      if (queue_list_[i]->task_size_ != 0) {
        succ_flag = false;
        break;
      }
    }
  }
  MS_LOG(INFO) << "Finish " << tasks.size() << " task successful";
  return CheckResult();
}

bool ThreadPool::CheckResult() {
  bool succ_flag = true;
  for (auto result : error_info_) {
    if (result.second.first) {
      MS_LOG(ERROR) << "task " << result.first << " failed, error code is " << result.second.second;
      succ_flag = false;
    }
  }
  return succ_flag;
}

ThreadPool *ThreadPool::GetInstance() {
  static ThreadPool instance;
  return &instance;
}

ThreadPool::~ThreadPool() {
  cur_thread_run_nums_ = static_cast<int>(thread_list_.size());
  exit_run_ = true;
  SubRunThread(0);
  queue_ready_.notify_all();
  for (auto &it : thread_list_) {
    if (it.joinable()) {
      it.join();
    }
  }
  for (const auto &it : activate_list_) {
    delete it;
  }
}
}  // namespace mindspore

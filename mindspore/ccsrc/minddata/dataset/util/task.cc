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
#include "minddata/dataset/util/task.h"
#include "utils/ms_utils.h"
#include "minddata/dataset/util/log_adapter.h"
#include "minddata/dataset/util/task_manager.h"
#if defined(__ANDROID__) || defined(ANDROID)
#include "minddata/dataset/util/services.h"
#endif

namespace mindspore {
namespace dataset {
thread_local Task *gMyTask = nullptr;

void Task::operator()() {
#if !defined(_WIN32) && !defined(_WIN64)
  gMyTask = this;
#endif
  id_ = this_thread::get_id();
  std::stringstream ss;
  ss << id_;
#if defined(__ANDROID__) || defined(ANDROID)
  // The thread id in Linux may be duplicate
  ss << Services::GetUniqueID();
#endif
  MS_LOG(DEBUG) << my_name_ << " Thread ID " << ss.str() << " Started.";

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID)
  native_handle_ = pthread_self();
#endif

  try {
    // Previously there is a timing hole where the thread is spawn but hit error immediately before we can set
    // the TaskGroup pointer and register. We move the registration logic to here (after we spawn) so we can
    // get the thread id.
    TaskGroup *vg = MyTaskGroup();
    rc_ = vg->GetIntrpService()->Register(ss.str(), this);
    if (rc_.IsOk()) {
      // Now we can run the given task.
      rc_ = fnc_obj_();
    }
    // Some error codes are ignored, e.g. interrupt. Others we just shutdown the group.
    if (rc_.IsError() && !rc_.IsInterrupted()) {
      ShutdownGroup();
    }
  } catch (const std::bad_alloc &e) {
    rc_ = Status(StatusCode::kOutOfMemory, __LINE__, __FILE__, e.what());
    ShutdownGroup();
  } catch (const std::exception &e) {
    rc_ = Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, e.what());
    ShutdownGroup();
  }
}

void Task::ShutdownGroup() {  // Wake up watch dog and shutdown the engine.
  {
    std::lock_guard<std::mutex> lk(mux_);
    caught_severe_exception_ = true;
  }
  TaskGroup *vg = MyTaskGroup();
  // If multiple threads hit severe errors in the same group. Keep the first one and
  // discard the rest.
  if (vg->rc_.IsOk()) {
    std::unique_lock<std::mutex> rcLock(vg->rc_mux_);
    // Check again after we get the lock
    if (vg->rc_.IsOk()) {
      vg->rc_ = rc_;
      rcLock.unlock();
      TaskManager::InterruptMaster(rc_);
      TaskManager::InterruptGroup(*this);
    }
  }
}

Status Task::GetTaskErrorIfAny() const {
  std::lock_guard<std::mutex> lk(mux_);
  if (caught_severe_exception_) {
    return rc_;
  } else {
    return Status::OK();
  }
}

Task::Task(const std::string &myName, const std::function<Status()> &f)
    : my_name_(myName),
      rc_(),
      fnc_obj_(f),
      task_group_(nullptr),
      is_master_(false),
      running_(false),
      caught_severe_exception_(false),
      native_handle_(0) {
  IntrpResource::ResetIntrpState();
  wp_.ResetIntrpState();
  wp_.Clear();
}

Status Task::Run() {
  Status rc;
  if (running_ == false) {
    try {
      thrd_ = std::async(std::launch::async, std::ref(*this));
      running_ = true;
      caught_severe_exception_ = false;
    } catch (const std::exception &e) {
      rc = Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, e.what());
    }
  }
  return rc;
}

Status Task::Join(WaitFlag blocking) {
  if (running_) {
    RETURN_UNEXPECTED_IF_NULL(MyTaskGroup());
    auto interrupt_svc = MyTaskGroup()->GetIntrpService();
    try {
      if (blocking == WaitFlag::kBlocking) {
        // If we are asked to wait, then wait
        thrd_.get();
      } else if (blocking == WaitFlag::kNonBlocking) {
        // There is a race condition in the global resource tracking such that a thread can miss the
        // interrupt and becomes blocked on a conditional variable forever. As a result, calling
        // join() will not come back. We need some timeout version of join such that if the thread
        // doesn't come back in a reasonable of time, we will send the interrupt again.
        while (thrd_.wait_for(std::chrono::seconds(1)) != std::future_status::ready) {
          // We can't tell which conditional_variable this thread is waiting on. So we may need
          // to interrupt everything one more time.
          MS_LOG(INFO) << "Some threads not responding. Interrupt again";
          interrupt_svc->InterruptAll();
        }
      } else {
        RETURN_STATUS_UNEXPECTED("Unknown WaitFlag");
      }
      std::stringstream ss;
      ss << get_id();
      MS_LOG(DEBUG) << MyName() << " Thread ID " << ss.str() << " Stopped.";
      running_ = false;
      RETURN_IF_NOT_OK(wp_.Deregister());
      RETURN_IF_NOT_OK(interrupt_svc->Deregister(ss.str()));
    } catch (const std::exception &e) {
      RETURN_STATUS_UNEXPECTED(e.what());
    }
  }
  return Status::OK();
}

TaskGroup *Task::MyTaskGroup() { return task_group_; }

void Task::set_task_group(TaskGroup *vg) { task_group_ = vg; }

Task::~Task() { task_group_ = nullptr; }
Status Task::OverrideInterruptRc(const Status &rc) {
  if (rc.IsInterrupted() && this_thread::is_master_thread()) {
    // If we are interrupted, override the return value if this is the master thread.
    // Master thread is being interrupted mostly because of some thread is reporting error.
    return TaskManager::GetMasterThreadRc();
  }
  return rc;
}

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && !defined(ANDROID)
pthread_t Task::GetNativeHandle() const { return native_handle_; }
#endif

}  // namespace dataset
}  // namespace mindspore

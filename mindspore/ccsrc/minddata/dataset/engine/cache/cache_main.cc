/**
 * Copyright 2020 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include "minddata/dataset/engine/cache/cache_server.h"
#include <sys/types.h>
#include <unistd.h>
#ifdef USE_GLOG
#include <glog/logging.h>
#endif
#include <cstdlib>
#include <thread>
#include <chrono>
#include "minddata/dataset/engine/cache/cache_common.h"
#include "minddata/dataset/engine/cache/cache_ipc.h"
namespace ds = mindspore::dataset;

/// Start the server
/// \param argv
/// \return Status object
ds::Status StartServer(int argc, char **argv) {
  ds::Status rc;
  ds::CacheServer::Builder builder;
  if (argc != 8) {
    return ds::Status(ds::StatusCode::kSyntaxError);
  }

  int32_t port = strtol(argv[3], nullptr, 10);
  builder.SetRootDirectory(argv[1])
    .SetNumWorkers(strtol(argv[2], nullptr, 10))
    .SetPort(port)
    .SetSharedMemorySizeInGB(strtol(argv[4], nullptr, 10))
    .SetMemoryCapRatio(strtof(argv[7], nullptr));

#ifdef USE_GLOG
  FLAGS_minloglevel = strtol(argv[5], nullptr, 10);
#endif

  auto daemonize_string = argv[6];
  bool daemonize = strcmp(daemonize_string, "true") == 0 || strcmp(daemonize_string, "TRUE") == 0 ||
                   strcmp(daemonize_string, "t") == 0 || strcmp(daemonize_string, "T") == 0;

  // We always change directory to / on unix rather than using the directory where the cache_server
  // is called. This is a standard procedure for daemonize a process on unix.
  if (chdir("/") == -1) {
    std::string errMsg = "Unable to change directory to /. Errno = " + std::to_string(errno);
    return ds::Status(ds::StatusCode::kUnexpectedError, __LINE__, __FILE__, errMsg);
  }

  // A message queue for communication between parent and child (if we fork).
  ds::SharedMessage msg;
  if (daemonize) {
#ifdef USE_GLOG
    FLAGS_log_dir = "/tmp";
    google::InitGoogleLogging(argv[0]);
#endif
    rc = msg.Create();
    if (rc.IsError()) {
      return rc;
    }
    pid_t pid = fork();
    // failed to fork
    if (pid < 0) {
      std::string errMsg = "Failed to fork process for cache server. Errno = " + std::to_string(errno);
      return ds::Status(ds::StatusCode::kUnexpectedError, __LINE__, __FILE__, errMsg);
    } else if (pid > 0) {
      // Parent and will be responsible for remove the queue on exit.
      msg.RemoveResourcesOnExit();
      // Sleep one second and we attach to the msg que
      std::this_thread::sleep_for(std::chrono::seconds(1));
      ds::Status child_rc;
      rc = msg.ReceiveStatus(&child_rc);
      if (rc.IsError()) {
        return rc;
      }
      if (child_rc.IsError()) {
        return child_rc;
      }
      std::cerr << "cache server daemon process has been created as process id: " << pid
                << "\nCheck log file for any start up error" << std::endl;
      signal(SIGCHLD, SIG_IGN);  // ignore sig child signal.
      return ds::Status::OK();
    } else {
      // Child process will continue from here if daemonize and parent has already exited.
      // If we are running in the foreground, none of the code in block below will be run.
      pid_t sid;
      umask(0);
      sid = setsid();
      if (sid < 0) {
        std::string errMsg = "Failed to setsid(). Errno = " + std::to_string(errno);
        return ds::Status(ds::StatusCode::kUnexpectedError, __LINE__, __FILE__, errMsg);
      }
      close(0);
      close(1);
      close(2);
    }
  }

  // Dump the summary
  MS_LOG(INFO) << builder << std::endl;
  // Create the instance with some sanity checks built in
  rc = builder.Build();
  if (rc.IsOk()) {
    // If all goes well, kick off the threads. Loop forever and never return unless error.
    ds::CacheServer &cs = ds::CacheServer::GetInstance();
    rc = cs.Run(msg.GetMsgQueueId());
  } else if (daemonize) {
    // If we didn't pass the sanity check to at least create the instance, use
    // the message queue to return the error message if this is the child daemon.
    return msg.SendStatus(rc);
  }
  return rc;
}

int main(int argc, char **argv) {
  // This executable is not to be called directly, and should be invoked by cache_admin executable.
  ds::Status rc = StartServer(argc, argv);
  // Check result
  if (rc.IsError()) {
    auto errCode = rc.get_code();
    auto errMsg = rc.ToString();
    std::cerr << errMsg << std::endl;
    return static_cast<int>(errCode);
  }
  return 0;
}

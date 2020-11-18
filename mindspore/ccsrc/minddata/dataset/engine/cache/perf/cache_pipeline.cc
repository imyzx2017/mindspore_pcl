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

#ifdef USE_GLOG
#include <glog/logging.h>
#endif
#include <string.h>
#include "minddata/dataset/engine/cache/perf/cache_pipeline_run.h"
namespace ds = mindspore::dataset;

int main(int argc, char **argv) {
#ifdef USE_GLOG
  FLAGS_log_dir = "/tmp";
  FLAGS_minloglevel = google::WARNING;
  google::InitGoogleLogging(argv[0]);
#endif
  ds::CachePipelineRun cachePipelineRun;
  if (cachePipelineRun.ProcessArgs(argc, argv) == 0) {
    ds::Status rc = cachePipelineRun.Run();
    // If we hit any error, send the rc back to the parent.
    if (rc.IsError()) {
      ds::ErrorMsg proto;
      proto.set_rc(static_cast<int32_t>(rc.get_code()));
      proto.set_msg(rc.ToString());
      ds::CachePerfMsg msg;
      (void)cachePipelineRun.SendMessage(&msg, ds::CachePerfMsg::MessageType::kError, &proto);
    }
    return static_cast<int>(rc.get_code());
  }
  return 0;
}

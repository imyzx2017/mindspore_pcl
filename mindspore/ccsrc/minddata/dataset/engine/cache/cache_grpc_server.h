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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_GRPC_SERVER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_GRPC_SERVER_H_

#include <atomic>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "minddata/dataset/engine/cache/cache_common.h"
#include "minddata/dataset/engine/cache/cache_arena.h"
#include "minddata/dataset/util/allocator.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/util/task_manager.h"

namespace mindspore {
namespace dataset {
/// \brief Server side view of BaseRequest. Incoming request are in the form of protobuf objects
/// and this class is used to translate from protobuf to structures understood by CacheService class.
/// \see CacheService
class CacheServerRequest : public BaseRequest {
 public:
  friend class CacheServer;
  friend class CacheService;
  enum class STATE : int8_t { CREATE = 1, PROCESS = 2, FINISH = 3 };
  explicit CacheServerRequest(int32_t queue_id)
      : BaseRequest::BaseRequest(BaseRequest::RequestType::kRequestUnknown),
        qid_(queue_id),
        st_(STATE::CREATE),
        responder_(&ctx_) {}

  ~CacheServerRequest() override = default;

  /// \brief Functor. Used mainly by CacheServerGreeterImpl class to tag each incoming request and this
  /// functor will translate each protobuf into some form understood by by CacheService class.
  /// \param svc Async service
  /// \param cq Completion queue
  /// \return Status object
  Status operator()(CacheServerGreeter::AsyncService *svc, grpc::ServerCompletionQueue *cq);

  /// \brief Override the base class Print method
  /// \param out
  void Print(std::ostream &out) const override;

  /// \brief Getter of the queue id
  /// \return The queue where the request should go to
  int32_t getQid() const { return qid_; }

 private:
  int32_t qid_;
  Status rc_;
  STATE st_;
  grpc::ServerContext ctx_;
  grpc::ServerAsyncResponseWriter<CacheReply> responder_;
};

/// \brief Implementation of CacheServerGreeter
/// \note It is an async server
/// \see cache_grpc.proto
class CacheServerGreeterImpl final {
  friend class CacheServer;

 public:
  explicit CacheServerGreeterImpl(int32_t port, int32_t shared_memory_sz_in_gb);
  virtual ~CacheServerGreeterImpl();
  /// \brief Brings up gRPC server
  /// \return none
  Status Run();
  /// \brief Entry function to handle cache server request
  Status HandleRequest(int32_t worker_id);

  /// Return the shared memory pool.
  /// \return Return the shared memory pool
  CachedSharedMemoryArena *GetSharedMemoryPool() { return shm_pool_.get(); }

  /// \brief Montor the status of the unix socket in case it is gone.
  Status MonitorUnixSocket();

  void Shutdown();

 private:
  int32_t port_;
  size_t shm_pool_sz_in_gb_;
  std::string unix_socket_;
  CacheServerGreeter::AsyncService svc_;
  std::unique_ptr<grpc::ServerCompletionQueue> cq_;
  std::unique_ptr<grpc::Server> server_;
  std::unique_ptr<CachedSharedMemoryArena> shm_pool_;
  SharedMemory::shm_key_t shm_key_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_GRPC_SERVER_H_

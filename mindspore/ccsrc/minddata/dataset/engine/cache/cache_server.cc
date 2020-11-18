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
#include <algorithm>
#include <functional>
#include <limits>
#include <vector>
#include "minddata/dataset/core/constants.h"
#include "minddata/dataset/engine/cache/cache_ipc.h"
#include "minddata/dataset/engine/cache/cache_service.h"
#include "minddata/dataset/engine/cache/cache_request.h"
#include "minddata/dataset/util/bit.h"
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/random.h"
#ifdef CACHE_LOCAL_CLIENT
#include "minddata/dataset/util/sig_handler.h"
#endif

namespace mindspore {
namespace dataset {
CacheServer *CacheServer::instance_ = nullptr;
std::once_flag CacheServer::init_instance_flag_;
Status CacheServer::DoServiceStart() {
#ifdef CACHE_LOCAL_CLIENT
  // We need to destroy the shared memory if user hits Control-C
  RegisterHandlers();
#endif
  if (!top_.empty()) {
    Path spill(top_);
    RETURN_IF_NOT_OK(spill.CreateDirectories());
    MS_LOG(INFO) << "CacheServer will use disk folder: " << top_;
  }
  RETURN_IF_NOT_OK(vg_.ServiceStart());
  RETURN_IF_NOT_OK(hw_info_->GetNumaNodeInfo());
  auto num_numa_nodes = GetNumaNodeCount();
  // If we link with numa library. Set default memory policy.
  // If we don't pin thread to cpu, then use up all memory controllers to maximize
  // memory bandwidth.
  RETURN_IF_NOT_OK(
    CacheServerHW::SetDefaultMemoryPolicy(numa_affinity_ ? CachePoolPolicy::kLocal : CachePoolPolicy::kInterleave));
  auto my_node = hw_info_->GetMyNode();
  MS_LOG(DEBUG) << "Cache server is running on numa node " << my_node;
  // Bump up num_workers_ to at least the number of numa nodes
  num_workers_ = std::max(num_numa_nodes, num_workers_);
  // But also it shouldn't be too many more than the hardware concurrency
  auto num_cpus = hw_info_->GetCpuCount();
  num_workers_ = std::min(2 * num_cpus, num_workers_);
  // Round up num_workers to a multiple of numa nodes.
  auto remainder = num_workers_ % num_numa_nodes;
  if (remainder > 0) num_workers_ += (num_numa_nodes - remainder);
  MS_LOG(INFO) << "Re-adjusting the number of workers to " << num_workers_;
  // There will be some threads working on the grpc queue and
  // some number of threads working on the CacheServerRequest queue.
  // Like a connector object we will set up the same number of queues but
  // we do not need to preserve any order. We will set the capacity of
  // each queue to be 64 since we are just pushing memory pointers which
  // is only 8 byte each.
  const int32_t kQueCapacity = 64;
  // This is the request queue from the client
  cache_q_ = std::make_shared<QueueList<CacheServerRequest *>>();
  cache_q_->Init(num_workers_, kQueCapacity);
  // We will match the number of grpc workers with the number of server workers.
  // But technically they don't have to be the same.
  num_grpc_workers_ = num_workers_;
  MS_LOG(DEBUG) << "Number of gprc workers is set to " << num_grpc_workers_;
  // For the grpc completion queue to work, we need to allocate some
  // tags which in our case are instances of CacheServerQuest.
  // They got recycled and we will allocate them in advance and push
  // them into some free list. We need more (two or three times) the
  // size of the cache_q. While each worker is working on a CacheSerRequest,
  // we need some extra running injecting in the the qrpc completion queue.
  const int32_t kMultiplier = 2;
  int ratio = num_workers_ / num_grpc_workers_;
  if (num_workers_ % num_grpc_workers_) ++ratio;
  const int32_t free_list_capacity = kMultiplier * (kQueCapacity + 1) * ratio;
  free_list_ = std::make_shared<QueueList<CacheServerRequest *>>();
  free_list_->Init(num_grpc_workers_, free_list_capacity);
  tag_.reserve(num_grpc_workers_);
  // Now we populate all free list. Round robin the free list among the numa nodes.
  for (auto m = 0; m < num_grpc_workers_; ++m) {
    NumaAllocator<CacheServerRequest> alloc(m % num_numa_nodes, CachePoolPolicy::kPreferred);
    // Ideally we allocate all the free list in one malloc. But we will allocate one segment
    // at a time so that we can change the numa policy easily per grpc worker.
    auto my_tag = std::make_unique<MemGuard<CacheServerRequest, NumaAllocator<CacheServerRequest>>>(alloc);
    // Allocate the tag and assign it the current queue
    RETURN_IF_NOT_OK(my_tag->allocate(free_list_capacity, m));
    for (int i = 0; i < free_list_capacity; ++i) {
      RETURN_IF_NOT_OK(free_list_->operator[](m)->Add((*my_tag)[i]));
    }
    tag_.push_back(std::move(my_tag));
  }
  RETURN_IF_NOT_OK(cache_q_->Register(&vg_));
  RETURN_IF_NOT_OK(free_list_->Register(&vg_));
  // Start the comm layer
  try {
    comm_layer_ = std::make_shared<CacheServerGreeterImpl>(port_, shared_memory_sz_in_gb_);
    RETURN_IF_NOT_OK(comm_layer_->Run());
    // Bring up a thread to monitor the unix socket in case it is removed.
    auto inotify_f = std::bind(&CacheServerGreeterImpl::MonitorUnixSocket, comm_layer_.get());
    RETURN_IF_NOT_OK(vg_.CreateAsyncTask("Monitor unix socket", inotify_f));
  } catch (const std::exception &e) {
    RETURN_STATUS_UNEXPECTED(e.what());
  }
  // Spawn a few threads to serve the real request.
  auto f = std::bind(&CacheServer::ServerRequest, this, std::placeholders::_1);
  for (auto i = 0; i < num_workers_; ++i) {
    Task *pTask;
    RETURN_IF_NOT_OK(vg_.CreateAsyncTask("Cache service worker", std::bind(f, i), &pTask));
    // Save a copy of the pointer to the underlying Task object. We may dynamically change their affinity if needed.
    numa_tasks_.emplace(i, pTask);
    // Spread out all the threads to all the numa nodes if needed
    if (IsNumaAffinityOn()) {
      auto numa_id = i % num_numa_nodes;
      RETURN_IF_NOT_OK(SetAffinity(*pTask, numa_id));
    }
  }
  // Finally loop forever to handle the request.
  auto r = std::bind(&CacheServer::RpcRequest, this, std::placeholders::_1);
  for (auto i = 0; i < num_grpc_workers_; ++i) {
    Task *pTask;
    RETURN_IF_NOT_OK(vg_.CreateAsyncTask("rpc worker", std::bind(r, i), &pTask));
    // All these grpc workers will be allocated to the same node which is where we allocate all those free tag
    // memory.
    if (IsNumaAffinityOn()) {
      RETURN_IF_NOT_OK(SetAffinity(*pTask, i % num_numa_nodes));
    }
  }
  return Status::OK();
}

Status CacheServer::DoServiceStop() {
  Status rc;
  Status rc2;
  // First stop all the threads.
  RETURN_IF_NOT_OK(vg_.ServiceStop());
  // Clean up all the caches if any.
  UniqueLock lck(&rwLock_);
  auto it = all_caches_.begin();
  while (it != all_caches_.end()) {
    auto cs = std::move(it->second);
    rc2 = cs->ServiceStop();
    if (rc2.IsError()) {
      rc = rc2;
    }
    ++it;
  }
  // Also remove the path we use to generate ftok.
  Path p(PortToUnixSocketPath(port_));
  (void)p.Remove();
  // Finally wake up cache_admin if it is waiting
  for (int32_t qID : shutdown_qIDs_) {
    SharedMessage msg(qID);
    msg.RemoveResourcesOnExit();
    // Let msg goes out of scope which will destroy the queue.
  }
  return rc;
}

CacheService *CacheServer::GetService(connection_id_type id) const {
  auto it = all_caches_.find(id);
  if (it != all_caches_.end()) {
    return it->second.get();
  }
  return nullptr;
}

Status CacheServer::CreateService(CacheRequest *rq, CacheReply *reply) {
  CHECK_FAIL_RETURN_UNEXPECTED(rq->has_connection_info(), "Missing connection info");
  std::string cookie;
  int32_t client_id;
  auto session_id = rq->connection_info().session_id();
  auto crc = rq->connection_info().crc();

  // Before allowing the creation, make sure the session had already been created by the user
  // Our intention is to add this cache to the active sessions list so leave the list locked during
  // this entire function.
  UniqueLock sess_lck(&sessions_lock_);
  auto session_it = active_sessions_.find(session_id);
  if (session_it == active_sessions_.end()) {
    RETURN_STATUS_UNEXPECTED("A cache creation has been requested but the session was not found!");
  }

  // We concat both numbers to form the internal connection id.
  auto connection_id = GetConnectionID(session_id, crc);
  CHECK_FAIL_RETURN_UNEXPECTED(!rq->buf_data().empty(), "Missing info to create cache");
  auto &create_cache_buf = rq->buf_data(0);
  auto p = flatbuffers::GetRoot<CreateCacheRequestMsg>(create_cache_buf.data());
  auto flag = static_cast<CreateCacheRequest::CreateCacheFlag>(p->flag());
  auto cache_mem_sz = p->cache_mem_sz();
  // We can't do spilling unless this server is setup with a spill path in the first place
  bool spill =
    (flag & CreateCacheRequest::CreateCacheFlag::kSpillToDisk) == CreateCacheRequest::CreateCacheFlag::kSpillToDisk;
  bool generate_id =
    (flag & CreateCacheRequest::CreateCacheFlag::kGenerateRowId) == CreateCacheRequest::CreateCacheFlag::kGenerateRowId;
  if (spill && top_.empty()) {
    RETURN_STATUS_UNEXPECTED("Server is not set up with spill support.");
  }
  flatbuffers::FlatBufferBuilder fbb;
  flatbuffers::Offset<flatbuffers::String> off_cookie;
  flatbuffers::Offset<flatbuffers::Vector<cpu_id_t>> off_cpu_list;
  // Before creating the cache, first check if this is a request for a shared usage of an existing cache
  // If two CreateService come in with identical connection_id, we need to serialize the create.
  // The first create will be successful and be given a special cookie.
  UniqueLock lck(&rwLock_);
  // Early exit if we are doing global shutdown
  if (global_shutdown_) {
    return Status::OK();
  }
  // We would like to protect ourselves from over allocating too much. We will go over existing cache
  // and calculate how much we have consumed so far.
  auto end = all_caches_.end();
  auto it = all_caches_.begin();
  bool duplicate = false;
  auto avail_mem = CacheServerHW::GetTotalSystemMemory() * memory_cap_ratio_;
  int64_t max_avail = avail_mem;
  while (it != end) {
    if (it->first == connection_id) {
      duplicate = true;
      break;
    } else {
      auto &cs = it->second;
      CacheService::ServiceStat stat;
      RETURN_IF_NOT_OK(cs->GetStat(&stat));
      int64_t mem_consumed = stat.stat_.num_mem_cached * stat.stat_.average_cache_sz;
      max_avail -= mem_consumed;
      if (max_avail <= 0) {
        return Status(StatusCode::kOutOfMemory, __LINE__, __FILE__, "Please destroy some sessions");
      }
    }
    ++it;
  }
  if (it == end) {
    // If we have some cache using some memory already, make a reasonable decision if we should return
    // out of memory.
    if (max_avail < avail_mem) {
      int64_t req_mem = cache_mem_sz * 1048576L;  // It is in MB unit.
      if (req_mem > max_avail) {
        return Status(StatusCode::kOutOfMemory, __LINE__, __FILE__, "Please destroy some sessions");
      } else if (req_mem == 0) {
        // This cache request is specifying unlimited memory up to the memory cap. If we have consumed more than
        // 85% of our limit, fail this request.
        if (static_cast<float>(max_avail) / static_cast<float>(avail_mem) <= 0.15) {
          return Status(StatusCode::kOutOfMemory, __LINE__, __FILE__, "Please destroy some sessions");
        }
      }
    }
    std::unique_ptr<CacheService> cs;
    try {
      cs = std::make_unique<CacheService>(cache_mem_sz, spill ? top_ : "", generate_id);
      RETURN_IF_NOT_OK(cs->ServiceStart());
      cookie = cs->cookie();
      client_id = cs->num_clients_.fetch_add(1);
      all_caches_.emplace(connection_id, std::move(cs));
    } catch (const std::bad_alloc &e) {
      return Status(StatusCode::kOutOfMemory);
    }
  } else {
    duplicate = true;
    client_id = it->second->num_clients_.fetch_add(1);
    MS_LOG(INFO) << "Duplicate request for " + std::to_string(connection_id) + " to create cache service";
  }
  // Shuffle the worker threads. But we need to release the locks or we will deadlock when calling
  // the following function
  lck.Unlock();
  sess_lck.Unlock();
  auto numa_id = client_id % GetNumaNodeCount();
  std::vector<cpu_id_t> cpu_list = hw_info_->GetCpuList(numa_id);
  // Send back the data
  off_cookie = fbb.CreateString(cookie);
  off_cpu_list = fbb.CreateVector(cpu_list);
  CreateCacheReplyMsgBuilder bld(fbb);
  bld.add_connection_id(connection_id);
  bld.add_cookie(off_cookie);
  bld.add_client_id(client_id);
  // The last thing we send back is a set of cpu id that we suggest the client should bind itself to
  bld.add_cpu_id(off_cpu_list);
  auto off = bld.Finish();
  fbb.Finish(off);
  reply->set_result(fbb.GetBufferPointer(), fbb.GetSize());
  // We can return OK but we will return a duplicate key so user can act accordingly to either ignore it
  // treat it as OK.
  return duplicate ? Status(StatusCode::kDuplicateKey) : Status::OK();
}

Status CacheServer::DestroyCache(CacheRequest *rq) {
  // We need a strong lock to protect the map.
  UniqueLock lck(&rwLock_);
  auto id = rq->connection_id();
  CacheService *cs = GetService(id);
  // it is already destroyed. Ignore it.
  if (cs != nullptr) {
    MS_LOG(WARNING) << "Dropping cache with connection id " << std::to_string(id);
    // std::map will invoke the destructor of CacheService. So we don't need to do anything here.
    auto n = all_caches_.erase(id);
    if (n == 0) {
      // It has been destroyed by another duplicate request.
      MS_LOG(INFO) << "Duplicate request for " + std::to_string(id) + " to create cache service";
    }
  }
  // We aren't touching the session list even though we may be dropping the last remaining cache of a session.
  // Leave that to be done by the drop session command.
  return Status::OK();
}

Status CacheServer::CacheRow(CacheRequest *rq, CacheReply *reply) {
  auto connection_id = rq->connection_id();
  // Hold the shared lock to prevent the cache from being dropped.
  SharedLock lck(&rwLock_);
  CacheService *cs = GetService(connection_id);
  if (cs == nullptr) {
    std::string errMsg = "Cache id " + std::to_string(connection_id) + " not found";
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, errMsg);
  } else {
    auto sz = rq->buf_data_size();
    std::vector<const void *> buffers;
    buffers.reserve(sz);
    // First piece of data is the cookie and is required
    CHECK_FAIL_RETURN_UNEXPECTED(!rq->buf_data().empty(), "Missing cookie");
    auto &cookie = rq->buf_data(0);
    // Only if the cookie matches, we can accept insert into this cache that has a build phase
    if (!cs->HasBuildPhase() || cookie == cs->cookie()) {
      // Push the address of each buffer (in the form of std::string coming in from protobuf) into
      // a vector of buffer
      for (auto i = 1; i < sz; ++i) {
        buffers.push_back(rq->buf_data(i).data());
      }
      row_id_type id = -1;
      // We will allocate the memory the same numa node this thread is bound to.
      RETURN_IF_NOT_OK(cs->CacheRow(buffers, &id));
      reply->set_result(std::to_string(id));
    } else {
      return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, "Cookie mismatch");
    }
  }
  return Status::OK();
}

Status CacheServer::FastCacheRow(CacheRequest *rq, CacheReply *reply) {
  auto connection_id = rq->connection_id();
  // Hold the shared lock to prevent the cache from being dropped.
  SharedLock lck(&rwLock_);
  CacheService *cs = GetService(connection_id);
  auto shared_pool = comm_layer_->GetSharedMemoryPool();
  auto *base = shared_pool->SharedMemoryBaseAddr();
  // Ensure we got 3 pieces of data coming in
  CHECK_FAIL_RETURN_UNEXPECTED(rq->buf_data_size() == 3, "Incomplete data");
  // First piece of data is the cookie and is required
  auto &cookie = rq->buf_data(0);
  // Second piece of data is the address where we can find the serialized data
  auto addr = strtoll(rq->buf_data(1).data(), nullptr, 10);
  auto p = reinterpret_cast<void *>(reinterpret_cast<int64_t>(base) + addr);
  // Third piece of data is the size of the serialized data that we need to transfer
  auto sz = strtoll(rq->buf_data(2).data(), nullptr, 10);
  // Successful or not, we need to free the memory on exit.
  Status rc;
  if (cs == nullptr) {
    std::string errMsg = "Cache id " + std::to_string(connection_id) + " not found";
    rc = Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, errMsg);
  } else {
    // Only if the cookie matches, we can accept insert into this cache that has a build phase
    if (!cs->HasBuildPhase() || cookie == cs->cookie()) {
      row_id_type id = -1;
      ReadableSlice src(p, sz);
      // We will allocate the memory the same numa node this thread is bound to.
      rc = cs->FastCacheRow(src, &id);
      reply->set_result(std::to_string(id));
    } else {
      rc = Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, "Cookie mismatch");
    }
  }
  // Return the block to the shared memory.
  shared_pool->Deallocate(p);
  return rc;
}

Status CacheServer::BatchFetch(const std::shared_ptr<flatbuffers::FlatBufferBuilder> &fbb, WritableSlice *out) {
  RETURN_UNEXPECTED_IF_NULL(out);
  int32_t numQ = GetNumGrpcWorkers();
  auto rng = GetRandomDevice();
  std::uniform_int_distribution<session_id_type> distribution(0, numQ - 1);
  int32_t qID = distribution(rng);
  std::vector<CacheServerRequest *> cache_rq_list;
  auto p = flatbuffers::GetRoot<BatchDataLocatorMsg>(fbb->GetBufferPointer());
  const auto num_elements = p->rows()->size();
  auto connection_id = p->connection_id();
  cache_rq_list.reserve(num_elements);
  int64_t data_offset = (num_elements + 1) * sizeof(int64_t);
  auto *offset_array = reinterpret_cast<int64_t *>(out->GetMutablePointer());
  offset_array[0] = data_offset;
  for (auto i = 0; i < num_elements; ++i) {
    auto data_locator = p->rows()->Get(i);
    auto node_id = data_locator->node_id();
    size_t sz = data_locator->size();
    void *source_addr = reinterpret_cast<void *>(data_locator->addr());
    auto key = data_locator->key();
    // Please read the comment in CacheServer::BatchFetchRows where we allocate
    // the buffer big enough so each thread (which we are going to dispatch) will
    // not run into false sharing problem. We are going to round up sz to 4k.
    auto sz_4k = round_up_4K(sz);
    offset_array[i + 1] = offset_array[i] + sz_4k;
    if (sz > 0) {
      WritableSlice row_data(*out, offset_array[i], sz);
      // Get a request and send to the proper worker (at some numa node) to do the fetch.
      worker_id_t worker_id = IsNumaAffinityOn() ? GetWorkerByNumaId(node_id) : GetRandomWorker();
      CacheServerRequest *cache_rq;
      RETURN_IF_NOT_OK(GetFreeRequestTag(qID++ % numQ, &cache_rq));
      cache_rq_list.push_back(cache_rq);
      // Set up all the necessarily field.
      cache_rq->type_ = BaseRequest::RequestType::kInternalFetchRow;
      cache_rq->st_ = CacheServerRequest::STATE::PROCESS;
      cache_rq->rq_.set_connection_id(connection_id);
      cache_rq->rq_.set_type(static_cast<int16_t>(cache_rq->type_));
      auto dest_addr = row_data.GetMutablePointer();
      flatbuffers::FlatBufferBuilder fb2;
      FetchRowMsgBuilder bld(fb2);
      bld.add_key(key);
      bld.add_size(sz);
      bld.add_source_addr(reinterpret_cast<int64_t>(source_addr));
      bld.add_dest_addr(reinterpret_cast<int64_t>(dest_addr));
      auto offset = bld.Finish();
      fb2.Finish(offset);
      cache_rq->rq_.add_buf_data(fb2.GetBufferPointer(), fb2.GetSize());
      RETURN_IF_NOT_OK(PushRequest(worker_id, cache_rq));
    }
  }
  // Now wait for all of them to come back.
  Status rc;
  for (CacheServerRequest *rq : cache_rq_list) {
    RETURN_IF_NOT_OK(rq->Wait());
    if (rq->rc_.IsError() && !rq->rc_.IsInterrupted() && rc.IsOk()) {
      rc = rq->rc_;
    }
    RETURN_IF_NOT_OK(ReturnRequestTag(rq));
  }
  return rc;
}

Status CacheServer::BatchFetchRows(CacheRequest *rq, CacheReply *reply) {
  auto connection_id = rq->connection_id();
  // Hold the shared lock to prevent the cache from being dropped.
  SharedLock lck(&rwLock_);
  CacheService *cs = GetService(connection_id);
  if (cs == nullptr) {
    std::string errMsg = "Cache id " + std::to_string(connection_id) + " not found";
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, errMsg);
  } else {
    CHECK_FAIL_RETURN_UNEXPECTED(!rq->buf_data().empty(), "Missing row id");
    auto &row_id_buf = rq->buf_data(0);
    auto p = flatbuffers::GetRoot<TensorRowIds>(row_id_buf.data());
    std::vector<row_id_type> row_id;
    auto sz = p->row_id()->size();
    row_id.reserve(sz);
    for (auto i = 0; i < sz; ++i) {
      row_id.push_back(p->row_id()->Get(i));
    }
    std::shared_ptr<flatbuffers::FlatBufferBuilder> fbb = std::make_shared<flatbuffers::FlatBufferBuilder>();
    RETURN_IF_NOT_OK(cs->PreBatchFetch(connection_id, row_id, fbb));
    // Let go of the shared lock. We don't need to interact with the CacheService anymore.
    // We shouldn't be holding any lock while we can wait for a long time for the rows to come back.
    lck.Unlock();
    auto locator = flatbuffers::GetRoot<BatchDataLocatorMsg>(fbb->GetBufferPointer());
    int64_t mem_sz = sizeof(int64_t) * (sz + 1);
    for (auto i = 0; i < sz; ++i) {
      auto row_sz = locator->rows()->Get(i)->size();
      // row_sz is the size of the cached data. Later we will spawn multiple threads
      // each of which will copy the data into either shared memory or protobuf concurrently but
      // to different region.
      // To avoid false sharing, we will bump up row_sz to be a multiple of 4k, i.e. 4096 bytes
      row_sz = round_up_4K(row_sz);
      mem_sz += row_sz;
    }
    auto client_flag = rq->flag();
    bool local_client = BitTest(client_flag, kLocalClientSupport);
    // For large amount data to be sent back, we will use shared memory provided it is a local
    // client that has local bypass support
    bool local_bypass = local_client ? (mem_sz >= kLocalByPassThreshold) : false;
    reply->set_flag(local_bypass ? kDataIsInSharedMemory : 0);
    if (local_bypass) {
      // We will use shared memory
      auto shared_pool = comm_layer_->GetSharedMemoryPool();
      auto *base = shared_pool->SharedMemoryBaseAddr();
      void *q = nullptr;
      RETURN_IF_NOT_OK(shared_pool->Allocate(mem_sz, &q));
      WritableSlice dest(q, mem_sz);
      Status rc = BatchFetch(fbb, &dest);
      if (rc.IsError()) {
        shared_pool->Deallocate(q);
        return rc;
      }
      // We can't return the absolute address which makes no sense to the client.
      // Instead we return the difference.
      auto difference = reinterpret_cast<int64_t>(q) - reinterpret_cast<int64_t>(base);
      reply->set_result(std::to_string(difference));
    } else {
      // We are going to use std::string to allocate and hold the result which will be eventually
      // 'moved' to the protobuf message (which underneath is also a std::string) for the purpose
      // to minimize memory copy.
      std::string mem;
      try {
        mem.resize(mem_sz);
        CHECK_FAIL_RETURN_UNEXPECTED(mem.capacity() >= mem_sz, "Programming error");
      } catch (const std::bad_alloc &e) {
        return Status(StatusCode::kOutOfMemory);
      }
      WritableSlice dest(mem.data(), mem_sz);
      RETURN_IF_NOT_OK(BatchFetch(fbb, &dest));
      reply->set_result(std::move(mem));
    }
  }
  return Status::OK();
}

Status CacheServer::GetStat(CacheRequest *rq, CacheReply *reply) {
  auto connection_id = rq->connection_id();
  // Hold the shared lock to prevent the cache from being dropped.
  SharedLock lck(&rwLock_);
  CacheService *cs = GetService(connection_id);
  if (cs == nullptr) {
    std::string errMsg = "Connection " + std::to_string(connection_id) + " not found";
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, errMsg);
  } else {
    CacheService::ServiceStat svc_stat;
    RETURN_IF_NOT_OK(cs->GetStat(&svc_stat));
    flatbuffers::FlatBufferBuilder fbb;
    ServiceStatMsgBuilder bld(fbb);
    bld.add_num_disk_cached(svc_stat.stat_.num_disk_cached);
    bld.add_num_mem_cached(svc_stat.stat_.num_mem_cached);
    bld.add_avg_cache_sz(svc_stat.stat_.average_cache_sz);
    bld.add_num_numa_hit(svc_stat.stat_.num_numa_hit);
    bld.add_max_row_id(svc_stat.stat_.max_key);
    bld.add_min_row_id(svc_stat.stat_.min_key);
    bld.add_state(svc_stat.state_);
    auto offset = bld.Finish();
    fbb.Finish(offset);
    reply->set_result(fbb.GetBufferPointer(), fbb.GetSize());
  }
  return Status::OK();
}

Status CacheServer::CacheSchema(CacheRequest *rq) {
  auto connection_id = rq->connection_id();
  // Hold the shared lock to prevent the cache from being dropped.
  SharedLock lck(&rwLock_);
  CacheService *cs = GetService(connection_id);
  if (cs == nullptr) {
    std::string errMsg = "Connection " + std::to_string(connection_id) + " not found";
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, errMsg);
  } else {
    CHECK_FAIL_RETURN_UNEXPECTED(!rq->buf_data().empty(), "Missing schema information");
    auto &create_schema_buf = rq->buf_data(0);
    RETURN_IF_NOT_OK(cs->CacheSchema(create_schema_buf.data(), create_schema_buf.size()));
  }
  return Status::OK();
}

Status CacheServer::FetchSchema(CacheRequest *rq, CacheReply *reply) {
  auto connection_id = rq->connection_id();
  // Hold the shared lock to prevent the cache from being dropped.
  SharedLock lck(&rwLock_);
  CacheService *cs = GetService(connection_id);
  if (cs == nullptr) {
    std::string errMsg = "Connection " + std::to_string(connection_id) + " not found";
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, errMsg);
  } else {
    // We are going to use std::string to allocate and hold the result which will be eventually
    // 'moved' to the protobuf message (which underneath is also a std::string) for the purpose
    // to minimize memory copy.
    std::string mem;
    RETURN_IF_NOT_OK(cs->FetchSchema(&mem));
    reply->set_result(std::move(mem));
  }
  return Status::OK();
}

Status CacheServer::BuildPhaseDone(CacheRequest *rq) {
  auto connection_id = rq->connection_id();
  // Hold the shared lock to prevent the cache from being dropped.
  SharedLock lck(&rwLock_);
  CacheService *cs = GetService(connection_id);
  if (cs == nullptr) {
    std::string errMsg = "Connection " + std::to_string(connection_id) + " not found";
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, errMsg);
  } else {
    // First piece of data is the cookie
    CHECK_FAIL_RETURN_UNEXPECTED(!rq->buf_data().empty(), "Missing cookie");
    auto &cookie = rq->buf_data(0);
    // We can only allow to switch phase is the cookie match.
    if (cookie == cs->cookie()) {
      RETURN_IF_NOT_OK(cs->BuildPhaseDone());
    } else {
      return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, "Cookie mismatch");
    }
  }
  return Status::OK();
}

Status CacheServer::GetCacheMissKeys(CacheRequest *rq, CacheReply *reply) {
  auto connection_id = rq->connection_id();
  // Hold the shared lock to prevent the cache from being dropped.
  SharedLock lck(&rwLock_);
  CacheService *cs = GetService(connection_id);
  if (cs == nullptr) {
    std::string errMsg = "Connection " + std::to_string(connection_id) + " not found";
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, errMsg);
  } else {
    std::vector<row_id_type> gap;
    RETURN_IF_NOT_OK(cs->FindKeysMiss(&gap));
    flatbuffers::FlatBufferBuilder fbb;
    auto off_t = fbb.CreateVector(gap);
    TensorRowIdsBuilder bld(fbb);
    bld.add_row_id(off_t);
    auto off = bld.Finish();
    fbb.Finish(off);
    reply->set_result(fbb.GetBufferPointer(), fbb.GetSize());
  }
  return Status::OK();
}

inline Status GenerateClientSessionID(session_id_type session_id, CacheReply *reply) {
  reply->set_result(std::to_string(session_id));
  return Status::OK();
}

Status CacheServer::ToggleWriteMode(CacheRequest *rq) {
  auto connection_id = rq->connection_id();
  // Hold the shared lock to prevent the cache from being dropped.
  SharedLock lck(&rwLock_);
  CacheService *cs = GetService(connection_id);
  if (cs == nullptr) {
    std::string errMsg = "Connection " + std::to_string(connection_id) + " not found";
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, errMsg);
  } else {
    // First piece of data is the on/off flag
    CHECK_FAIL_RETURN_UNEXPECTED(!rq->buf_data().empty(), "Missing action flag");
    const auto &action = rq->buf_data(0);
    bool on_off = false;
    if (strcmp(action.data(), "on") == 0) {
      on_off = true;
    } else if (strcmp(action.data(), "off") == 0) {
      on_off = false;
    } else {
      RETURN_STATUS_UNEXPECTED("Unknown request: " + action);
    }
    RETURN_IF_NOT_OK(cs->ToggleWriteMode(on_off));
  }
  return Status::OK();
}

Status CacheServer::ListSessions(CacheReply *reply) {
  SharedLock sess_lck(&sessions_lock_);
  SharedLock lck(&rwLock_);
  flatbuffers::FlatBufferBuilder fbb;
  std::vector<flatbuffers::Offset<ListSessionMsg>> session_msgs_vector;
  for (auto const &current_session_id : active_sessions_) {
    bool found = false;
    for (auto const &it : all_caches_) {
      auto current_conn_id = it.first;
      if (GetSessionID(current_conn_id) == current_session_id) {
        found = true;
        auto &cs = it.second;
        CacheService::ServiceStat svc_stat;
        RETURN_IF_NOT_OK(cs->GetStat(&svc_stat));
        auto current_stats = CreateServiceStatMsg(fbb, svc_stat.stat_.num_mem_cached, svc_stat.stat_.num_disk_cached,
                                                  svc_stat.stat_.average_cache_sz, svc_stat.stat_.num_numa_hit,
                                                  svc_stat.stat_.min_key, svc_stat.stat_.max_key, svc_stat.state_);
        auto current_session_info = CreateListSessionMsg(fbb, current_session_id, current_conn_id, current_stats);
        session_msgs_vector.push_back(current_session_info);
      }
    }
    if (!found) {
      // If there is no cache created yet, assign a connection id of 0 along with empty stats
      auto current_stats = CreateServiceStatMsg(fbb, 0, 0, 0, 0, 0, 0);
      auto current_session_info = CreateListSessionMsg(fbb, current_session_id, 0, current_stats);
      session_msgs_vector.push_back(current_session_info);
    }
  }
  auto session_msgs = fbb.CreateVector(session_msgs_vector);
  ListSessionsMsgBuilder s_builder(fbb);
  s_builder.add_sessions(session_msgs);
  auto offset = s_builder.Finish();
  fbb.Finish(offset);
  reply->set_result(fbb.GetBufferPointer(), fbb.GetSize());
  return Status::OK();
}

Status CacheServer::ConnectReset(CacheRequest *rq) {
  auto connection_id = rq->connection_id();
  // Hold the shared lock to prevent the cache from being dropped.
  SharedLock lck(&rwLock_);
  CacheService *cs = GetService(connection_id);
  if (cs == nullptr) {
    std::string errMsg = "Connection " + std::to_string(connection_id) + " not found";
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, errMsg);
  } else {
    auto client_id = rq->client_id();
    MS_LOG(WARNING) << "Client id " << client_id << " with connection id " << connection_id << " disconnects";
    cs->num_clients_--;
  }
  return Status::OK();
}

/// \brief This is the main loop the cache server thread(s) are running.
/// Each thread will pop a request and send the result back to the client using grpc
/// \return
Status CacheServer::ServerRequest(worker_id_t worker_id) {
  TaskManager::FindMe()->Post();
  MS_LOG(DEBUG) << "Worker id " << worker_id << " is running on node " << hw_info_->GetMyNode();
  auto &my_que = cache_q_->operator[](worker_id);
  // Loop forever until we are interrupted or shutdown.
  while (!global_shutdown_) {
    bool internal_request = false;
    CacheServerRequest *cache_req = nullptr;
    RETURN_IF_NOT_OK(my_que->PopFront(&cache_req));
    auto &rq = cache_req->rq_;
    auto &reply = cache_req->reply_;
    // Except for creating a new session, we expect cs is not null.
    switch (cache_req->type_) {
      case BaseRequest::RequestType::kCacheRow: {
        // Look into the flag to see where we can find the data and
        // call the appropriate method.
        auto flag = rq.flag();
        if (BitTest(flag, kDataIsInSharedMemory)) {
          cache_req->rc_ = FastCacheRow(&rq, &reply);
        } else {
          cache_req->rc_ = CacheRow(&rq, &reply);
        }
        break;
      }
      case BaseRequest::RequestType::kInternalFetchRow: {
        internal_request = true;
        auto connection_id = rq.connection_id();
        SharedLock lck(&rwLock_);
        CacheService *cs = GetService(connection_id);
        if (cs == nullptr) {
          std::string errMsg = "Connection " + std::to_string(connection_id) + " not found";
          cache_req->rc_ = Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, errMsg);
        } else {
          cache_req->rc_ = cs->InternalFetchRow(flatbuffers::GetRoot<FetchRowMsg>(rq.buf_data(0).data()));
        }
        break;
      }
      case BaseRequest::RequestType::kCreateCache: {
        cache_req->rc_ = CreateService(&rq, &reply);
        break;
      }
      case BaseRequest::RequestType::kGetCacheMissKeys: {
        cache_req->rc_ = GetCacheMissKeys(&rq, &reply);
        break;
      }
      case BaseRequest::RequestType::kDestroyCache: {
        cache_req->rc_ = DestroyCache(&rq);
        break;
      }
      case BaseRequest::RequestType::kGetStat: {
        cache_req->rc_ = GetStat(&rq, &reply);
        break;
      }
      case BaseRequest::RequestType::kCacheSchema: {
        cache_req->rc_ = CacheSchema(&rq);
        break;
      }
      case BaseRequest::RequestType::kFetchSchema: {
        cache_req->rc_ = FetchSchema(&rq, &reply);
        break;
      }
      case BaseRequest::RequestType::kBuildPhaseDone: {
        cache_req->rc_ = BuildPhaseDone(&rq);
        break;
      }
      case BaseRequest::RequestType::kDropSession: {
        cache_req->rc_ = DestroySession(&rq);
        break;
      }
      case BaseRequest::RequestType::kGenerateSessionId: {
        cache_req->rc_ = GenerateClientSessionID(GenerateSessionID(), &reply);
        break;
      }
      case BaseRequest::RequestType::kAllocateSharedBlock: {
        cache_req->rc_ = AllocateSharedMemory(&rq, &reply);
        break;
      }
      case BaseRequest::RequestType::kFreeSharedBlock: {
        cache_req->rc_ = FreeSharedMemory(&rq);
        break;
      }
      case BaseRequest::RequestType::kStopService: {
        // This command shutdowns everything.
        cache_req->rc_ = GlobalShutdown(cache_req);
        break;
      }
      case BaseRequest::RequestType::kHeartBeat: {
        cache_req->rc_ = Status::OK();
        break;
      }
      case BaseRequest::RequestType::kToggleWriteMode: {
        cache_req->rc_ = ToggleWriteMode(&rq);
        break;
      }
      case BaseRequest::RequestType::kListSessions: {
        cache_req->rc_ = ListSessions(&reply);
        break;
      }
      case BaseRequest::RequestType::kConnectReset: {
        cache_req->rc_ = ConnectReset(&rq);
        break;
      }
      default:
        std::string errMsg("Unknown request type : ");
        errMsg += std::to_string(static_cast<uint16_t>(cache_req->type_));
        cache_req->rc_ = Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, errMsg);
    }
    // Notify it is done, and move on to the next request.
    Status2CacheReply(cache_req->rc_, &reply);
    cache_req->st_ = CacheServerRequest::STATE::FINISH;
    // We will re-tag the request back to the grpc queue. Once it comes back from the client,
    // the CacheServerRequest, i.e. the pointer cache_req, will be free
    if (!global_shutdown_) {
      if (!internal_request) {
        cache_req->responder_.Finish(reply, grpc::Status::OK, cache_req);
      } else {
        // This is an internal request and is not tied to rpc. But need to post because there
        // is a thread waiting on the completion of this request.
        cache_req->wp_.Set();
      }
    }
  }
  return Status::OK();
}

connection_id_type CacheServer::GetConnectionID(session_id_type session_id, uint32_t crc) const {
  connection_id_type connection_id =
    (static_cast<connection_id_type>(session_id) << 32u) | static_cast<connection_id_type>(crc);
  return connection_id;
}

session_id_type CacheServer::GetSessionID(connection_id_type connection_id) const {
  return static_cast<session_id_type>(connection_id >> 32u);
}

CacheServer::CacheServer(const std::string &spill_path, int32_t num_workers, int32_t port,
                         int32_t shared_meory_sz_in_gb, float memory_cap_ratio)
    : top_(spill_path),
      num_workers_(num_workers),
      num_grpc_workers_(num_workers_),
      port_(port),
      shared_memory_sz_in_gb_(shared_meory_sz_in_gb),
      global_shutdown_(false),
      memory_cap_ratio_(memory_cap_ratio),
      numa_affinity_(true) {
  hw_info_ = std::make_shared<CacheServerHW>();
  // If we are not linked with numa library (i.e. NUMA_ENABLED is false), turn off cpu
  // affinity which can make performance worse.
  if (!CacheServerHW::numa_enabled()) {
    numa_affinity_ = false;
    MS_LOG(WARNING) << "Warning: This build is not compiled with numa support.  Install libnuma-devel and use a build "
                       "that is compiled with numa support for more optimal performance";
  }
}

Status CacheServer::Run(int msg_qid) {
  Status rc = ServiceStart();
  // If there is a message que, return the status now before we call join_all which will never return
  if (msg_qid != -1) {
    SharedMessage msg(msg_qid);
    RETURN_IF_NOT_OK(msg.SendStatus(rc));
  }
  if (rc.IsError()) {
    return rc;
  }
  // This is called by the main function and we shouldn't exit. Otherwise the main thread
  // will just shutdown. So we will call some function that never return unless error.
  // One good case will be simply to wait for all threads to return.
  // note that after we have sent the initial status using the msg_qid, parent process will exit and
  // remove it. So we can't use it again.
  RETURN_IF_NOT_OK(vg_.join_all(Task::WaitFlag::kBlocking));
  return Status::OK();
}

Status CacheServer::GetFreeRequestTag(int32_t queue_id, CacheServerRequest **q) {
  RETURN_UNEXPECTED_IF_NULL(q);
  CacheServer &cs = CacheServer::GetInstance();
  CacheServerRequest *p;
  RETURN_IF_NOT_OK(cs.free_list_->operator[](queue_id)->PopFront(&p));
  *q = p;
  return Status::OK();
}

Status CacheServer::ReturnRequestTag(CacheServerRequest *p) {
  RETURN_UNEXPECTED_IF_NULL(p);
  int32_t myQID = p->getQid();
  // Free any memory from the protobufs
  p->~CacheServerRequest();
  // Re-initialize the memory
  new (p) CacheServerRequest(myQID);
  // Now we return it back to free list.
  CacheServer &cs = CacheServer::GetInstance();
  RETURN_IF_NOT_OK(cs.free_list_->operator[](myQID)->Add(p));
  return Status::OK();
}

Status CacheServer::DestroySession(CacheRequest *rq) {
  CHECK_FAIL_RETURN_UNEXPECTED(rq->has_connection_info(), "Missing session id");
  auto drop_session_id = rq->connection_info().session_id();
  // Grab the locks in the correct order to avoid deadlock.
  UniqueLock sess_lck(&sessions_lock_);
  UniqueLock lck(&rwLock_);
  // Iterate over the set of connection id's for this session that we're dropping and erase each one.
  bool found = false;
  for (auto it = all_caches_.begin(); it != all_caches_.end();) {
    auto connection_id = it->first;
    auto session_id = GetSessionID(connection_id);
    // We can just call DestroyCache() but we are holding a lock already. Doing so will cause deadlock.
    // So we will just manually do it.
    if (session_id == drop_session_id) {
      found = true;
      it = all_caches_.erase(it);
      MS_LOG(INFO) << "Destroy cache with id " << connection_id;
    } else {
      ++it;
    }
  }
  // Finally remove the session itself
  auto n = active_sessions_.erase(drop_session_id);
  if (n > 0) {
    MS_LOG(INFO) << "Session destroyed with id " << drop_session_id;
    return Status::OK();
  } else {
    if (found) {
      std::string errMsg =
        "A destroy cache request has been completed but it had a stale session id " + std::to_string(drop_session_id);
      RETURN_STATUS_UNEXPECTED(errMsg);
    } else {
      std::string errMsg = "Session id " + std::to_string(drop_session_id) + " not found.";
      return Status(StatusCode::kFileNotExist, errMsg);
    }
  }
}

session_id_type CacheServer::GenerateSessionID() {
  UniqueLock sess_lck(&sessions_lock_);
  auto mt = GetRandomDevice();
  std::uniform_int_distribution<session_id_type> distribution(0, std::numeric_limits<session_id_type>::max());
  session_id_type session_id;
  bool duplicate = false;
  do {
    session_id = distribution(mt);
    auto r = active_sessions_.insert(session_id);
    duplicate = !r.second;
  } while (duplicate);
  return session_id;
}

Status CacheServer::AllocateSharedMemory(CacheRequest *rq, CacheReply *reply) {
  auto requestedSz = strtoll(rq->buf_data(0).data(), nullptr, 10);
  auto shared_pool = comm_layer_->GetSharedMemoryPool();
  auto *base = shared_pool->SharedMemoryBaseAddr();
  void *p = nullptr;
  RETURN_IF_NOT_OK(shared_pool->Allocate(requestedSz, &p));
  // We can't return the absolute address which makes no sense to the client.
  // Instead we return the difference.
  auto difference = reinterpret_cast<int64_t>(p) - reinterpret_cast<int64_t>(base);
  reply->set_result(std::to_string(difference));
  return Status::OK();
}

Status CacheServer::FreeSharedMemory(CacheRequest *rq) {
  auto shared_pool = comm_layer_->GetSharedMemoryPool();
  auto *base = shared_pool->SharedMemoryBaseAddr();
  auto addr = strtoll(rq->buf_data(0).data(), nullptr, 10);
  auto p = reinterpret_cast<void *>(reinterpret_cast<int64_t>(base) + addr);
  shared_pool->Deallocate(p);
  return Status::OK();
}

Status CacheServer::RpcRequest(worker_id_t worker_id) {
  TaskManager::FindMe()->Post();
  RETURN_IF_NOT_OK(comm_layer_->HandleRequest(worker_id));
  return Status::OK();
}

Status CacheServer::GlobalShutdown(CacheServerRequest *cache_req) {
  auto *rq = &cache_req->rq_;
  auto *reply = &cache_req->reply_;
  if (!rq->buf_data().empty()) {
    // cache_admin sends us a message qID and we will destroy the
    // queue in our destructor and this will wake up cache_admin.
    // But we don't want the cache_admin blindly just block itself.
    // So we will send back an ack before shutdown the comm layer.
    try {
      int32_t qID = std::stoi(rq->buf_data(0));
      shutdown_qIDs_.push_back(qID);
    } catch (const std::exception &e) {
      // ignore it.
    }
  }
  reply->set_result("OK");
  Status2CacheReply(cache_req->rc_, reply);
  cache_req->st_ = CacheServerRequest::STATE::FINISH;
  cache_req->responder_.Finish(*reply, grpc::Status::OK, cache_req);
  // Let's shutdown in proper order.
  bool expected = false;
  if (global_shutdown_.compare_exchange_strong(expected, true)) {
    MS_LOG(WARNING) << "Shutting down server.";
    // Shutdown the grpc queue. No longer accept any new comer.
    // The threads we spawn to work on the grpc queue will exit themselves once
    // they notice the queue has been shutdown.
    comm_layer_->Shutdown();
    // Now we interrupt any threads that are waiting on cache_q_
    vg_.interrupt_all();
    // The next thing to do drop all the caches.
    UniqueLock lck(&rwLock_);
    for (auto it = all_caches_.begin(); it != all_caches_.end();) {
      auto id = it->first;
      MS_LOG(WARNING) << "Dropping cache with connection id " << std::to_string(id);
      // Wait for all outstanding work to be finished.
      auto &cs = it->second;
      UniqueLock cs_lock(&cs->rw_lock_);
      it = all_caches_.erase(it);
    }
  }
  return Status::OK();
}

worker_id_t CacheServer::GetWorkerByNumaId(numa_id_t numa_id) const {
  auto num_numa_nodes = GetNumaNodeCount();
  MS_ASSERT(numa_id < num_numa_nodes);
  auto num_workers_per_node = GetNumWorkers() / num_numa_nodes;
  std::mt19937 gen = GetRandomDevice();
  std::uniform_int_distribution<worker_id_t> dist(0, num_workers_per_node - 1);
  auto n = dist(gen);
  worker_id_t worker_id = n * num_numa_nodes + numa_id;
  MS_ASSERT(worker_id < GetNumWorkers());
  return worker_id;
}

worker_id_t CacheServer::GetRandomWorker() const {
  std::mt19937 gen = GetRandomDevice();
  std::uniform_int_distribution<worker_id_t> dist(0, num_workers_ - 1);
  return dist(gen);
}

Status CacheServer::Builder::IpcResourceCleanup() {
  Status rc;
  SharedMemory::shm_key_t shm_key;
  auto unix_socket = PortToUnixSocketPath(port_);
  rc = PortToFtok(port_, &shm_key);
  // We are expecting the unix path doesn't exist.
  if (rc.IsError()) {
    return Status::OK();
  }
  // Attach to the shared memory which we expect don't exist
  SharedMemory mem(shm_key);
  rc = mem.Attach();
  if (rc.IsError()) {
    return Status::OK();
  } else {
    RETURN_IF_NOT_OK(mem.Detach());
  }
  int32_t num_attached;
  RETURN_IF_NOT_OK(mem.GetNumAttached(&num_attached));
  if (num_attached == 0) {
    // Stale shared memory from last time.
    // Remove both the memory and the socket path
    RETURN_IF_NOT_OK(mem.Destroy());
    Path p(unix_socket);
    (void)p.Remove();
  } else {
    // Server is already up.
    std::string errMsg = "Cache server is already up and running";
    // We return a duplicate error. The main() will intercept
    // and output a proper message
    return Status(StatusCode::kDuplicateKey, errMsg);
  }
  return Status::OK();
}

Status CacheServer::Builder::SanityCheck() {
  if (shared_memory_sz_in_gb_ <= 0) {
    RETURN_STATUS_UNEXPECTED("Shared memory size (in GB unit) must be positive");
  }
  if (num_workers_ <= 0) {
    RETURN_STATUS_UNEXPECTED("Number of parallel workers must be positive");
  }
  if (!top_.empty()) {
    auto p = top_.data();
    if (p[0] != '/') {
      RETURN_STATUS_UNEXPECTED("Spilling directory must be an absolute path");
    }
    // Check if the spill directory is writable
    Path spill(top_);
    auto t = spill / Services::GetUniqueID();
    Status rc = t.CreateDirectory();
    if (rc.IsOk()) {
      rc = t.Remove();
    }
    if (rc.IsError()) {
      RETURN_STATUS_UNEXPECTED("Spilling directory is not writable\n" + rc.ToString());
    }
  }
  if (memory_cap_ratio_ <= 0 || memory_cap_ratio_ > 1) {
    RETURN_STATUS_UNEXPECTED("Memory cap ratio should be positive and no greater than 1");
  }

  // Check if the shared memory.
  RETURN_IF_NOT_OK(IpcResourceCleanup());
  return Status::OK();
}

CacheServer::Builder::Builder()
    : top_("/tmp"),
      num_workers_(std::thread::hardware_concurrency() / 2),
      port_(50052),
      shared_memory_sz_in_gb_(4),
      memory_cap_ratio_(0.8) {
  if (num_workers_ == 0) {
    num_workers_ = 1;
  }
}
}  // namespace dataset
}  // namespace mindspore

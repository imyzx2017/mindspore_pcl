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

#ifndef MINDSPORE_CCSRC_PS_CORE_HTTP_SERVER_H_
#define MINDSPORE_CCSRC_PS_CORE_HTTP_SERVER_H_

#include "ps/core/http_message_handler.h"

#include <event2/buffer.h>
#include <event2/event.h>
#include <event2/http.h>
#include <event2/keyvalq_struct.h>
#include <event2/listener.h>
#include <event2/util.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>

namespace mindspore {
namespace ps {
namespace core {

typedef enum eHttpMethod {
  HM_GET = 1 << 0,
  HM_POST = 1 << 1,
  HM_HEAD = 1 << 2,
  HM_PUT = 1 << 3,
  HM_DELETE = 1 << 4,
  HM_OPTIONS = 1 << 5,
  HM_TRACE = 1 << 6,
  HM_CONNECT = 1 << 7,
  HM_PATCH = 1 << 8
} HttpMethod;

using OnRequestReceive = std::function<void(std::shared_ptr<HttpMessageHandler>)>;

class HttpServer {
 public:
  // Server address only support IPV4 now, and should be in format of "x.x.x.x"
  explicit HttpServer(const std::string &address, std::uint16_t port)
      : server_address_(address), server_port_(port), event_base_(nullptr), event_http_(nullptr), is_init_(false) {}

  ~HttpServer();

  bool InitServer();
  void SetTimeOut(int seconds = 5);

  // Default allowed methods: GET, POST, HEAD, PUT, DELETE
  void SetAllowedMethod(u_int16_t methods);

  // Default to ((((unsigned long long)0xffffffffUL) << 32) | 0xffffffffUL)
  void SetMaxHeaderSize(std::size_t num);

  // Default to ((((unsigned long long)0xffffffffUL) << 32) | 0xffffffffUL)
  void SetMaxBodySize(std::size_t num);

  // Return: true if success, false if failed, check log to find failure reason
  bool RegisterRoute(const std::string &url, OnRequestReceive *func);
  bool UnRegisterRoute(const std::string &url);

  bool Start();
  void Stop();

 private:
  std::string server_address_;
  std::uint16_t server_port_;
  struct event_base *event_base_;
  struct evhttp *event_http_;
  bool is_init_;
};

}  // namespace core
}  // namespace ps
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PS_CORE_HTTP_SERVER_H_

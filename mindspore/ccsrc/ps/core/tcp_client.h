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

#ifndef MINDSPORE_CCSRC_PS_CORE_TCP_CLIENT_H_
#define MINDSPORE_CCSRC_PS_CORE_TCP_CLIENT_H_

#include "ps/core/tcp_message_handler.h"

#include <event2/event.h>
#include <event2/bufferevent.h>
#include <functional>
#include <string>
#include <memory>
#include <vector>

#include "proto/comm.pb.h"
#include "ps/core/cluster_config.h"

namespace mindspore {
namespace ps {
namespace core {

class TcpClient {
 public:
  using OnConnected = std::function<void(const TcpClient &)>;
  using OnDisconnected = std::function<void(const TcpClient &, int)>;
  using OnRead = std::function<void(const TcpClient &, const void *, size_t)>;
  using OnTimeout = std::function<void(const TcpClient &)>;
  using OnMessage = std::function<void(const TcpClient &, const CommMessage &)>;

  explicit TcpClient(const std::string &address, std::uint16_t port);
  virtual ~TcpClient();

  std::string GetServerAddress() const;
  void SetCallback(const OnConnected &conn, const OnDisconnected &disconn, const OnRead &read,
                   const OnTimeout &timeout);
  void Init();
  void StartWithDelay(int seconds);
  void Stop();
  void Start();
  void StartWithNoBlock();
  void SetMessageCallback(const OnMessage &cb);
  void SendMessage(const CommMessage &message) const;
  void SendMessageWithTimer();

 protected:
  static void SetTcpNoDelay(const evutil_socket_t &fd);
  static void TimeoutCallback(evutil_socket_t fd, std::int16_t what, void *arg);
  static void ReadCallback(struct bufferevent *bev, void *ctx);
  static void EventCallback(struct bufferevent *bev, std::int16_t events, void *ptr);
  virtual void OnReadHandler(const void *buf, size_t num);
  static void SendHeartBeatCallback(evutil_socket_t fd, int16_t event, void *arg);

 private:
  OnMessage message_callback_;
  TcpMessageHandler message_handler_;

  OnConnected connected_callback_;
  OnDisconnected disconnected_callback_;
  OnRead read_callback_;
  OnTimeout timeout_callback_;

  event_base *event_base_;
  event *event_timeout_;
  bufferevent *buffer_event_;

  std::string server_address_;
  std::uint16_t server_port_;
};

}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_TCP_CLIENT_H_

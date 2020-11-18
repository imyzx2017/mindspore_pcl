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

#include "ps/core/tcp_message_handler.h"

#include <arpa/inet.h>
#include <iostream>
#include <utility>

namespace mindspore {
namespace ps {
namespace core {

void TcpMessageHandler::SetCallback(const messageReceive &message_receive) { message_callback_ = message_receive; }

void TcpMessageHandler::ReceiveMessage(const void *buffer, size_t num) {
  MS_EXCEPTION_IF_NULL(buffer);
  auto buffer_data = reinterpret_cast<const unsigned char *>(buffer);

  while (num > 0) {
    if (remaining_length_ == 0) {
      for (int i = 0; i < 4 && num > 0; ++i) {
        header_[++header_index_] = *(buffer_data + i);
        --num;
        if (header_index_ == 3) {
          message_length_ = *reinterpret_cast<const uint32_t *>(header_);
          remaining_length_ = message_length_;
          message_buffer_.reset(new unsigned char[remaining_length_]);
          buffer_data += (i + 1);
          break;
        }
      }
    }

    if (remaining_length_ > 0 && num > 0) {
      uint32_t copy_len = remaining_length_ <= num ? remaining_length_ : num;
      remaining_length_ -= copy_len;
      num -= copy_len;

      int ret = memcpy_s(message_buffer_.get() + last_copy_len_, copy_len, buffer_data, copy_len);
      last_copy_len_ += copy_len;
      buffer_data += copy_len;
      if (ret != 0) {
        MS_LOG(EXCEPTION) << "The memcpy_s error, errorno(" << ret << ")";
      }

      if (remaining_length_ == 0) {
        CommMessage pb_message;
        pb_message.ParseFromArray(message_buffer_.get(), message_length_);
        if (message_callback_) {
          message_callback_(pb_message);
        }
        message_buffer_.reset();
        message_buffer_ = nullptr;
        header_index_ = -1;
        last_copy_len_ = 0;
      }
    }
  }
}

}  // namespace core
}  // namespace ps
}  // namespace mindspore

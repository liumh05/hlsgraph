// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace hls {
template <typename T> class stream {
 public:
  stream() = default;
  explicit stream(const char*) {}
  void write(const T& value) { value_ = value; }
  T read() { return value_; }

 private:
  T value_{};
};
}  // namespace hls

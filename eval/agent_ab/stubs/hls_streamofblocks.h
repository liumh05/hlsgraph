// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace hls {
template <typename Block> class stream_of_blocks {};

template <typename Block> class read_lock;
template <typename Element, unsigned Size> class read_lock<Element[Size]> {
 public:
  explicit read_lock(stream_of_blocks<Element[Size]>&) {}
  const Element& operator[](unsigned index) const { return storage_[index]; }

 private:
  Element storage_[Size]{};
};

template <typename Block> class write_lock;
template <typename Element, unsigned Size> class write_lock<Element[Size]> {
 public:
  explicit write_lock(stream_of_blocks<Element[Size]>&) {}
  Element& operator[](unsigned index) { return storage_[index]; }

 private:
  Element storage_[Size]{};
};
}  // namespace hls

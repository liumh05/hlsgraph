// SPDX-License-Identifier: Apache-2.0
#pragma once

// Parser-only arithmetic surface. This header intentionally supplies no HLS
// implementation, scheduling, precision, or QoR semantics.
template <int Width, int IntegerBits> class ap_fixed {
 public:
  ap_fixed() = default;
  ap_fixed(double value) : value_(value) {}
  ap_fixed(int value) : value_(value) {}
  ap_fixed& operator=(double value) { value_ = value; return *this; }
  bool operator<(int rhs) const { return value_ < rhs; }
  ap_fixed operator+(const ap_fixed& rhs) const { return value_ + rhs.value_; }
  ap_fixed operator-(const ap_fixed& rhs) const { return value_ - rhs.value_; }
  ap_fixed operator*(const ap_fixed& rhs) const { return value_ * rhs.value_; }
  ap_fixed operator*(int rhs) const { return value_ * rhs; }
  ap_fixed operator/(int rhs) const { return value_ / rhs; }

 private:
  double value_ = 0.0;
};

template <int W, int I>
ap_fixed<W, I> operator*(int lhs, const ap_fixed<W, I>& rhs) {
  return rhs * lhs;
}

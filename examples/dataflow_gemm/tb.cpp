// SPDX-License-Identifier: Apache-2.0
#include <cassert>

void dut(const int input[16], int output[16]);

int main() {
  int input[16]{};
  int output[16]{};
  for (int i = 0; i < 16; ++i) input[i] = i;
  dut(input, output);
  for (int i = 0; i < 16; ++i) assert(output[i] == i * (i + 1));
  return 0;
}


// SPDX-License-Identifier: Apache-2.0
#include "hls_stream.h"

static void load(const int input[16], hls::stream<int>& values) {
load_loop:
  for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE II=1
    values.write(input[i]);
  }
}

static void compute(hls::stream<int>& values, hls::stream<int>& results) {
  int weights[16];
#pragma HLS ARRAY_PARTITION variable=weights cyclic factor=4
compute_loop:
  for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE II=2
    weights[i] = i + 1;
    results.write(values.read() * weights[i]);
  }
}

static void store(hls::stream<int>& results, int output[16]) {
store_loop:
  for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE II=1
    output[i] = results.read();
  }
}

void dut(const int input[16], int output[16]) {
#pragma HLS DATAFLOW
  hls::stream<int> values;
  hls::stream<int> results;
#pragma HLS STREAM variable=values depth=8
#pragma HLS STREAM variable=results depth=16
  load(input, values);
  compute(values, results);
  store(results, output);
}


// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "hls_stream.h"

namespace xf {
namespace database {
enum SortOrder { SORT_ASCENDING = 1, SORT_DESCENDING = 0 };

template <typename KeyType, int ParallelNumber>
void bitonicSort(hls::stream<KeyType>& input,
                 hls::stream<bool>& input_end,
                 hls::stream<KeyType>& output,
                 hls::stream<bool>& output_end,
                 bool order);
}  // namespace database
}  // namespace xf

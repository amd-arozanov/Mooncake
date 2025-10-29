// Copyright 2024 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "transport/nvlink_transport/stream_pool.h"
#include <glog/logging.h>

namespace mooncake {

StreamPool::StreamPool(size_t recommended_pool_size) : next_idx_(0) {
    int pool_size = recommended_pool_size;

    if (getenv("MC_NUM_GPU_STREAMS")) {
        try {
            pool_size = std::stoi(getenv("MC_NUM_GPU_STREAMS"));
        } catch (...) {
            LOG(ERROR) << "StreamPool: unable to read MC_NUM_GPU_STREAMS";
        }
    }

    for (size_t i = 0; i < pool_size; ++i) {
        cudaStream_t stream;

        cudaError_t result = cudaStreamCreate(&stream);
        if (result != cudaSuccess) {
            LOG(ERROR) << "StreamPool: unable to create stream (Error code: "
                       << result << " - " << cudaGetErrorString(result) << ")";
            break;
        }

        streams_.push_back(stream);
    }
}

StreamPool::~StreamPool() {
    for (auto s : streams_) {
        cudaError_t result = cudaStreamDestroy(s);
        if (result != cudaSuccess) {
            LOG(ERROR) << "StreamPool: unable to destroy stream (Error code: "
                       << result << " - " << cudaGetErrorString(result) << ")";
        }
    }
}

cudaStream_t StreamPool::getNextStream() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (streams_.size() == 0) return nullptr;

    cudaStream_t stream = streams_[next_idx_];
    next_idx_ = (next_idx_ + 1) % streams_.size();
    
    return stream;
}

} // namespace mooncake

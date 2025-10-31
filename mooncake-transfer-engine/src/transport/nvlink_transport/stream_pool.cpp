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

static bool checkError(cudaError_t result, const char* message) {
    if (result != cudaSuccess) {
        LOG(ERROR) << message << " (Error code: " << result << " - "
                   << cudaGetErrorString(result) << ")";
        return false;
    }
    return true;
}

StreamPool::StreamPool(size_t recommended_pool_size)
    : pool_size_(recommended_pool_size), num_devices_(0) {
    // Read pool size from environment variable if set
    if (getenv("MC_NUM_GPU_STREAMS")) {
        try {
            pool_size_ = std::stoull(getenv("MC_NUM_GPU_STREAMS"));
        } catch (...) {
            LOG(ERROR) << "StreamPool: unable to read MC_NUM_GPU_STREAMS, "
                       << "using default value " << recommended_pool_size;
            pool_size_ = recommended_pool_size;
        }
    }

    // Get number of devices and pre-allocate vector
    cudaError_t err = cudaGetDeviceCount(&num_devices_);
    if (!checkError(err, "StreamPool: failed to get device count")) {
        num_devices_ = 1;  // Fallback to single device
    }

    device_pools_.resize(num_devices_);
}

StreamPool::~StreamPool() {
    std::lock_guard<std::mutex> lock(global_mutex_);
    for (int device_id = 0; device_id < num_devices_; ++device_id) {
        auto& device_pool = device_pools_[device_id];
        std::lock_guard<std::mutex> device_lock(*device_pool.mutex_);
        for (auto s : device_pool.streams_) {
            cudaError_t result = cudaStreamDestroy(s);
            checkError(result, "StreamPool: unable to destroy stream");
        }
    }
}

void StreamPool::initializeDevicePool(int device_id) {
    if (device_id >= num_devices_) {
        LOG(ERROR) << "StreamPool: device_id " << device_id
                   << " exceeds available devices " << num_devices_;
        return;
    }

    // Set device before creating streams
    cudaError_t err = cudaSetDevice(device_id);
    if (!checkError(err, "StreamPool: failed to set device")) {
        return;
    }

    auto& device_pool = device_pools_[device_id];
    for (size_t i = 0; i < pool_size_; ++i) {
        cudaStream_t stream;

        cudaError_t result = cudaStreamCreate(&stream);
        if (!checkError(result, "StreamPool: unable to create stream")) {
            break;
        }

        device_pool.streams_.push_back(stream);
    }
    device_pool.initialized_ = true;
}

cudaStream_t StreamPool::getNextStream(int device_id) {
    // If device_id is not specified, get current device
    if (device_id == -1) {
        cudaError_t err = cudaGetDevice(&device_id);
        if (!checkError(err, "StreamPool: failed to get current device")) {
            return nullptr;
        }
    }

    if (device_id >= num_devices_) {
        LOG(ERROR) << "StreamPool: device_id " << device_id
                   << " exceeds available devices " << num_devices_;
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(global_mutex_);

    // Initialize device pool if it doesn't exist
    auto& device_pool = device_pools_[device_id];
    if (!device_pool.initialized_) {
        initializeDevicePool(device_id);
    }

    std::lock_guard<std::mutex> device_lock(*device_pool.mutex_);

    if (device_pool.streams_.size() == 0) return nullptr;

    cudaStream_t stream = device_pool.streams_[device_pool.next_idx_];
    device_pool.next_idx_ =
        (device_pool.next_idx_ + 1) % device_pool.streams_.size();

    return stream;
}

}  // namespace mooncake

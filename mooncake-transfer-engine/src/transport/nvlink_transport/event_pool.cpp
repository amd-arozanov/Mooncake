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

#include "transport/nvlink_transport/event_pool.h"
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

EventPool::EventPool(size_t pool_size)
    : pool_size_(pool_size), num_devices_(0) {
    // Get number of devices and pre-allocate vector
    cudaError_t err = cudaGetDeviceCount(&num_devices_);
    if (!checkError(err, "EventPool: failed to get device count")) {
        num_devices_ = 1;  // Fallback to single device
    }

    device_pools_.resize(num_devices_);
}

EventPool::~EventPool() {
    std::lock_guard<std::mutex> lock(global_mutex_);
    for (int device_id = 0; device_id < num_devices_; ++device_id) {
        auto& device_pool = device_pools_[device_id];
        std::lock_guard<std::mutex> device_lock(*device_pool.mutex_);
        for (auto e : device_pool.all_events_) {
            cudaError_t result = cudaEventDestroy(e);
            checkError(result, "EventPool: unable to destroy event");
        }
    }
}

void EventPool::initializeDevicePool(int device_id) {
    if (device_id >= num_devices_) {
        LOG(ERROR) << "EventPool: device_id " << device_id
                   << " exceeds available devices " << num_devices_;
        return;
    }

    // Set device before creating events
    cudaError_t err = cudaSetDevice(device_id);
    if (!checkError(err, "EventPool: failed to set device")) {
        return;
    }

    auto& device_pool = device_pools_[device_id];
    for (size_t i = 0; i < pool_size_; ++i) {
        cudaEvent_t event = createEvent(device_id);
        if (event != nullptr) {
            device_pool.available_events_.push_back(event);
            device_pool.all_events_.push_back(event);
        }
    }
    device_pool.initialized_ = true;
}

cudaEvent_t EventPool::getEvent(int device_id) {
    // If device_id is not specified, get current device
    if (device_id == -1) {
        cudaError_t err = cudaGetDevice(&device_id);
        if (!checkError(err, "EventPool: failed to get current device")) {
            return nullptr;
        }
    }

    if (device_id >= num_devices_) {
        LOG(ERROR) << "EventPool: device_id " << device_id
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

    // If no available events, create a new one
    if (device_pool.available_events_.empty()) {
        cudaEvent_t new_event = createEvent(device_id);
        if (new_event != nullptr) {
            device_pool.all_events_.push_back(new_event);
            return new_event;
        } else {
            LOG(ERROR) << "EventPool: unable to create new event when pool "
                       << "is empty for device " << device_id;
            return nullptr;
        }
    }

    // Get an available event from the back and remove it from available list
    cudaEvent_t event = device_pool.available_events_.back();
    device_pool.available_events_.pop_back();
    return event;
}

void EventPool::putEvent(cudaEvent_t event, int device_id) {
    // If device_id is not specified, get current device
    if (device_id == -1) {
        cudaError_t err = cudaGetDevice(&device_id);
        if (!checkError(err, "EventPool: failed to get current device")) {
            return;
        }
    }

    if (device_id >= num_devices_) {
        LOG(ERROR) << "EventPool: device_id " << device_id
                   << " exceeds available devices " << num_devices_;
        return;
    }

    std::lock_guard<std::mutex> lock(global_mutex_);

    auto& device_pool = device_pools_[device_id];
    if (!device_pool.initialized_) {
        LOG(ERROR) << "EventPool: trying to return event to "
                   << "non-initialized device pool " << device_id;
        return;
    }

    std::lock_guard<std::mutex> device_lock(*device_pool.mutex_);
    device_pool.available_events_.push_back(event);
}

cudaEvent_t EventPool::createEvent(int device_id) {
    // Set device before creating event
    cudaError_t err = cudaSetDevice(device_id);
    if (!checkError(err,
                    "EventPool: failed to set device when creating event")) {
        return nullptr;
    }

    cudaEvent_t event;

    cudaError_t result = cudaEventCreate(&event);
    if (!checkError(result, "EventPool: unable to create event")) {
        return nullptr;
    }

    return event;
}

}  // namespace mooncake

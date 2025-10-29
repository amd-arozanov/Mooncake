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

EventPool::EventPool(size_t pool_size)
{
    for (size_t i = 0; i < pool_size; ++i) {
        cudaEvent_t event = createEvent();
        if (event != nullptr) {
            available_events_.push_back(event);
            all_events_.push_back(event);
        }
    }
}

EventPool::~EventPool() 
{
    for (auto e : all_events_) {
        cudaError_t result = cudaEventDestroy(e);
        if (result != cudaSuccess) {
            LOG(ERROR) << "EventPool: unable to destroy event (Error code: "
                       << result << " - " << cudaGetErrorString(result) << ")";
        }
    }
}

cudaEvent_t EventPool::getEvent() 
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    // If no available events, create a new one
    if (available_events_.empty()) {
        cudaEvent_t new_event = createEvent();
        if (new_event != nullptr) {
            all_events_.push_back(new_event);
            return new_event;
        } else {
            LOG(ERROR) << "EventPool: unable to create new event when pool is empty";
            return nullptr;
        }
    }
    
    // Get an available event from the back and remove it from available list
    cudaEvent_t event = available_events_.back();
    available_events_.pop_back();
    return event;
}

void EventPool::putEvent(cudaEvent_t event) 
{
    std::lock_guard<std::mutex> lock(mutex_);
    available_events_.push_back(event);
}

cudaEvent_t EventPool::createEvent()
{
    cudaEvent_t event;
    
    cudaError_t result = cudaEventCreate(&event);
    if (result != cudaSuccess) {
        LOG(ERROR) << "EventPool: unable to create event (Error code: "
                   << result << " - " << cudaGetErrorString(result) << ")";
        return nullptr;
    }
    
    return event;
}

} // namespace mooncake
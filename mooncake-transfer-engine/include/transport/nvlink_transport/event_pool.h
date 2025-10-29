// Copyright 2024 KVCache.AI

#ifndef EVENT_POOL_H_
#define EVENT_POOL_H_

#ifdef USE_CUDA
#include <cuda_runtime.h>
#define event_t cudaEvent_t
#elif USE_ROCM
#include <hip/hip_runtime.h>
#define event_t hipEvent_t
#endif

#include <vector>
#include <mutex>

namespace mooncake {

class EventPool {
   public:
    EventPool(size_t pool_size);
    ~EventPool();

    event_t getEvent();
    void putEvent(event_t event);

   private:
    std::vector<event_t> available_events_;
    std::vector<event_t> all_events_;  // Track all events for cleanup
    std::mutex mutex_;
    
    // Helper method to create a new event
    event_t createEvent();
};

}  // namespace mooncake

#endif  // EVENT_POOL_H_

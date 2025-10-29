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
#include <memory>

namespace mooncake {

class EventPool {
   public:
    EventPool(size_t pool_size);
    ~EventPool();

    event_t getEvent(int device_id = -1);
    void putEvent(event_t event, int device_id = -1);

   private:
    struct DeviceEventPool {
        std::vector<event_t> available_events_;
        std::vector<event_t> all_events_;  // Track all events for cleanup
        std::unique_ptr<std::mutex> mutex_;
        bool initialized_;
        
        DeviceEventPool() : mutex_(new std::mutex()), initialized_(false) {}
    };
    
    std::vector<DeviceEventPool> device_pools_;
    std::mutex global_mutex_;
    size_t pool_size_;
    int num_devices_;
    
    void initializeDevicePool(int device_id);
    
    // Helper method to create a new event
    event_t createEvent(int device_id);
};

}  // namespace mooncake

#endif  // EVENT_POOL_H_

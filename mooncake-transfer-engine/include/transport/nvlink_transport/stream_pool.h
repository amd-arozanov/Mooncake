// Copyright 2024 KVCache.AI

#ifndef STREAM_POOL_H_
#define STREAM_POOL_H_

#include "cuda_alike.h"

#include <vector>
#include <mutex>
#include <memory>

namespace mooncake {

class StreamPool {
   public:
    StreamPool(size_t recommended_pool_size);
    ~StreamPool();

    cudaStream_t getNextStream(int device_id = -1);

   private:
    struct DeviceStreamPool {
        std::vector<cudaStream_t> streams_;
        size_t next_idx_;
        std::unique_ptr<std::mutex> mutex_;
        bool initialized_;

        DeviceStreamPool()
            : next_idx_(0), mutex_(new std::mutex()), initialized_(false) {}
    };

    std::vector<DeviceStreamPool> device_pools_;
    std::mutex global_mutex_;
    size_t pool_size_;
    int num_devices_;

    void initializeDevicePool(int device_id);
};

}  // namespace mooncake

#endif  // STREAM_POOL_H_

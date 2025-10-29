// Copyright 2024 KVCache.AI

#ifndef STREAM_POOL_H_
#define STREAM_POOL_H_

#ifdef USE_CUDA
#include <cuda_runtime.h>
#define stream_t cudaStream_t
#elif USE_ROCM
#include <hip/hip_runtime.h>
#define stream_t hipStream_t
#endif

#include <vector>
#include <mutex>

namespace mooncake {

class StreamPool {
   public:
    StreamPool(size_t recommended_pool_size);
    ~StreamPool();

    stream_t getNextStream();

   private:
    std::vector<stream_t> streams_;
    size_t next_idx_;
    std::mutex mutex_;
};

}  // namespace mooncake

#endif  // STREAM_POOL_H_
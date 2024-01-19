/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/stream_executor/gpu/gpu_cudamallocasync_allocator.h"

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#ifdef GOOGLE_CUDA
#define DRIVERVERSION 11030
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/cuda/cuda_activation.h"
#elif TENSORFLOW_USE_ROCM

#define DRIVERVERSION 50300
#include "xla/stream_executor/cuda/cuda_activation.h"
using cuuint64_t = uint64_t;
#endif  // GOOGLE_CUDA

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xla/stream_executor/gpu/gpu_init.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/framework/allocator.h"
#include "tsl/framework/device_id.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/mutex.h"
#include "tsl/util/env_var.h"

namespace stream_executor {

void GpuMallocAsyncAllocator::PrintAllocatorStatisticsNoLock() {
  std::map<size_t, int> size_map_histogram;
  std::vector<std::string> ptr_size_string;
  for (auto p : size_map_) {
    if (VLOG_IS_ON(8)) {
      ptr_size_string.push_back(
          absl::StrCat("(", absl::Hex(p.first), ",", p.second) + ")");
    }
    size_map_histogram[p.second]++;
  }
  LOG(ERROR) << "Histogram of current allocation: (allocation_size_in_bytes, "
             << "nb_allocation_of_that_sizes), ...;";
  for (auto p : size_map_histogram) {
    LOG(ERROR) << p.first << ", " << p.second;
  }

  VLOG(8) << "\nThe sorted list of (ptr,size):";
  VLOG(8) << absl::StrJoin(ptr_size_string, ",");

#if CUDA_VERSION >= DRIVERVERSION || TF_ROCM_VERSION >= DRIVERVERSION
  cuuint64_t mem_reserved_current;
  GpuDriver::GpuMemPoolGetAttribute(pool_, GpuDriver::MemPoolAttribute::kReservedMemCurrent, &mem_reserved_current)
  cuuint64_t mem_used_current;
  GpuDriver::GpuMemPoolGetAttribute(pool_, GpuDriver::MemPoolAttribute::kUsedMemCurrent, &mem_used_current);
  cuuint64_t mem_reserved_high;
  GpuDriver::GpuMemPoolGetAttribute(pool_, GpuDriver::MemPoolAttribute::kReservedMemHigh, &mem_reserved_high);
  cuuint64_t mem_used_high;
  GpuDriver::GpuMemPoolGetAttribute(pool_, GpuDriver::MemPoolAttribute::kUsedMemHigh, &mem_used_high);
  LOG(ERROR) << "CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT: " << mem_reserved_current;
  LOG(ERROR) << "CU_MEMPOOL_ATTR_USED_MEM_CURRENT: " << mem_used_current;
  LOG(ERROR) << "CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH: " << mem_reserved_high;
  LOG(ERROR) << "CU_MEMPOOL_ATTR_USED_MEM_HIGH: " << mem_used_high;
#endif
}

std::atomic<int> GpuMallocAsyncAllocator::number_instantiated_(0);

GpuMallocAsyncAllocator::GpuMallocAsyncAllocator(
    tsl::PlatformDeviceId platform_device_id, size_t pool_size,
    bool reserve_memory, bool compute_stats)
    : name_(absl::StrCat("gpu_async_", platform_device_id.value())),
      reserve_memory_(reserve_memory) {
  ++number_instantiated_;

  // Stop clang from complaining about unused private fields when
  // TF_CUDA_MALLOC_ASYNC_SUPPORTED is not defined.
  (void)reserve_memory_;

#if TF_GPU_MALLOC_ASYNC_SUPPORTED
  stream_exec_ = GPUMachineManager()
                     ->ExecutorForDevice(platform_device_id.value())
                     .value();
  // Initialized here as it only exist if compiled with a recent
  // enough CUDA.
  pool_ = nullptr;
  gpu_stream_ = nullptr;
  auto driverVersion = GpuDriver::GetDriverVersion().value();
  VLOG(2) << "DRIVER VERSION: " << driverVersion;
  if (driverVersion < 11020) {
    LOG(FATAL)  // Crash OK.
        << "Disable cuda_malloc_async or update your CUDA driver to a version"
        << " compatible with CUDA 11.2 or higher."
        << " We detected a version compatible with: " << driverVersion;
  }

  // WAR an CUDA 11.2 driver bug for multiple-GPU. It currently
  // request that the context on GPU 0 is initialized. Which isn't the
  // case for TF+horovod.
  if (platform_device_id.value() > 0 && driverVersion < DRIVERVERSION) {
    GpuContextHandle pctx;  // We loose track of it. But this is fine.
    auto result = GpuDriver::DevicePrimaryCtxRetain(static_cast<GpuDeviceHandle>(0))
	if (result){
      LOG(FATAL)  // Crash OK.
          << "Failed to retain context: " << result.message();
	} else {
	  pctx = results.value();
	}
  }

  // Check the CUDA runtime is recent enough.
  auto status2 = GpuDriver::GetDriverVersion();
  if (status2) {
    LOG(FATAL)  // Crash OK.
        << "Error while fetching driver version: "
        << status2.message();
  }

  // Check that cudaMallocAsync is supported.
  int gpu_malloc_async_supported;
  if (auto status =
          GpuDriver::DeviceGetAttribute(&gpu_malloc_async_supported,
                               CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,
                               platform_device_id.value())) {
    LOG(FATAL)  // Crash OK.
        << "On device: " << platform_device_id.value()
        << " Current driver: " << driverVersion
        << ". Failed to get device attribute : " << status.message();
  }
  if (!gpu_malloc_async_supported)
    LOG(FATAL)  // Crash OK.
        << "TF_GPU_ALLOCATOR=gpu_malloc_async isn't currently supported on "
        << "GPU id " << platform_device_id.value() << ":"
        << " Possible causes: device not supported (request SM60+), driver too "
           "old, "
        << " OS not supported, CUDA/ROCm version too old(request CUDA11.2+ or ROCm 5.3).";

  if (auto status =
          GpuDriver::DeviceGetDefaultMemPool(&pool_, platform_device_id.value()))
    LOG(FATAL) <<  // Crash OK.
        "Failed to get default CUDA pool: " << status.message();

  VLOG(1) << Name() << " CudaMallocAsync initialized on platform: "
          << platform_device_id.value() << " with pool size of: " << pool_size
          << " this ptr: " << this;
  uint64_t pool_size_64 = pool_size;
  if (auto status = cuMemPoolSetAttribute(
          pool_, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &pool_size_64))
    LOG(FATAL) <<  // Crash OK.
        "Failed to set CUDA pool attribute: " << GetCudaErrorMessage(status);

  if (compute_stats) {
    stats_ = std::make_unique<tsl::AllocatorStats>();
    stats_->bytes_limit = static_cast<int64_t>(pool_size);
  }  // If not set, it means we do not compute stats.

  // If in TF_DETERMINISTIC_ALLOCATOR is set, then make the allocator behave
  // determistically.
  bool deterministic = false;
  TF_CHECK_OK(tsl::ReadBoolFromEnvVar("TF_DETERMINISTIC_ALLOCATOR",
                                      /*default_val=*/false, &deterministic));
  if (deterministic) {
    int disable = 0;
    if (auto status = cuMemPoolSetAttribute(
            pool_, CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC, &disable)) {
      LOG(FATAL) <<  // Crash OK.
          "Failed to set CUDA pool attribute: " << GetCudaErrorMessage(status);
    }
    if (auto status = cuMemPoolSetAttribute(
            pool_, CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES,
            &disable)) {
      LOG(FATAL) <<  // Crash OK.
          "Failed to set CUDA pool attribute: " << GetCudaErrorMessage(status);
    }
  }

  // Set read/write access to all GPUs.
  static auto* all_pools_ = new std::vector<CUmemoryPool*>();
  static auto* all_ids_ = new std::vector<tsl::PlatformDeviceId>();
  DCHECK(all_pools_->size() == all_ids_->size());
  for (int i = 0; i < all_pools_->size(); ++i) {
    // Set the current pool access to the previous GPUs.
    CUmemAccessDesc map;
    map.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    map.location.id = (*all_ids_)[i].value();

    map.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    VLOG(2) << "Setting access of the current pool to "
            << " location id: " << map.location.id;
    int canAccessPeer;
    if (auto status = cuDeviceCanAccessPeer(
            &canAccessPeer, platform_device_id.value(), map.location.id)) {
      pool_ = nullptr;
      LOG(FATAL)  // Crash OK.
          << "cuDeviceCanAccessPeer failed to know if GPU id "
          << map.location.id << " can access GPU id "
          << platform_device_id.value() << ": " << GetCudaErrorMessage(status);
    }
    if (canAccessPeer == 1) {
      if (auto status = cuMemPoolSetAccess(pool_, &map, 1)) {
        pool_ = nullptr;
        LOG(FATAL)  // Crash OK.
            << "Error when setting access to the pool id: " << i
            << " location id: " << map.location.id
            << " error: " << GetCudaErrorMessage(status);
      }
    }

    // Set the previous pools access to the current GPU.
    map.location.id = platform_device_id.value();

    VLOG(2) << "Set access to the pool id: " << i
            << " location id: " << map.location.id;
    if (auto status = cuDeviceCanAccessPeer(&canAccessPeer, i,
                                            platform_device_id.value())) {
      pool_ = nullptr;
      LOG(FATAL)  // Crash OK.
          << "cuDeviceCanAccessPeer failed: " << GetCudaErrorMessage(status);
    }
    if (canAccessPeer == 1) {
      if (auto status = cuMemPoolSetAccess(*(*all_pools_)[i], &map, 1)) {
        pool_ = nullptr;
        LOG(FATAL)  // Crash OK.
            << "Error when setting access to the pool id: " << i
            << " location id: " << map.location.id
            << " error: " << GetCudaErrorMessage(status);
      }
    }
  }
  all_pools_->push_back(&pool_);
  all_ids_->push_back(platform_device_id);

  VLOG(2) << Name() << " GpuCudaMallocAsyncAllocator PoolSize " << pool_size;
#else   // TF_CUDA_MALLOC_ASYNC_SUPPORTED
  LOG(FATAL) << "GpuCudaMallocAsyncAllocator requires CUDA 11.2+";  // Crash OK.
#endif  // TF_CUDA_MALLOC_ASYNC_SUPPORTED
}

GpuCudaMallocAsyncAllocator::~GpuCudaMallocAsyncAllocator() {}

void* GpuCudaMallocAsyncAllocator::AllocateRaw(size_t alignment,
                                               size_t num_bytes) {
#if TF_CUDA_MALLOC_ASYNC_SUPPORTED
  CHECK(cuda_stream_ != nullptr)
      << "A stream must be added to the GpuCudaMallocAsync allocator";
  if (pool_ == nullptr) {
    LOG(FATAL)  // Crash OK.
        << "The instantiation of GpuCudaMallocAsyncAllocator failed."
        << " See previous errors.";
  }
  // The lock is only needed when stats are enabled, but it must be around
  // the cuMemAllocFromPoolAsync call as well to ensure consistency of the stats
  // update.
  std::unique_lock<tsl::mutex> lock(lock_, std::defer_lock);
  if (stats_) {
    lock.lock();
  }
  cuda::ScopedActivateExecutorContext scoped_activation{stream_exec_};
  void* ptr = nullptr;
  auto result = cuMemAllocFromPoolAsync(reinterpret_cast<CUdeviceptr*>(&ptr),
                                        num_bytes, pool_, cuda_stream_);
  if (result == CUDA_ERROR_OUT_OF_MEMORY) {
    // Doing a stream synchronization give the driver more flexibility
    // for blocks coalescing and doing memory remapping. So it can
    // solve some OOM cases when memory is tight.
    cuStreamSynchronize(cuda_stream_);
    result = cuMemAllocFromPoolAsync(reinterpret_cast<CUdeviceptr*>(&ptr),
                                     num_bytes, pool_, cuda_stream_);
  }
  if (result) {
    size_t free, total;
    cuMemGetInfo(&free, &total);
    LOG(ERROR) << Name() << " cuMemAllocAsync failed to allocate " << num_bytes
               << " bytes: " << GetCudaErrorMessage(result)
               << "\n Reported by CUDA: Free memory/Total memory: " << free
               << "/" << total;
    if (stats_) {
      LOG(ERROR) << "Stats: " << stats_->DebugString();
      PrintAllocatorStatisticsNoLock();
    }

    return nullptr;
  }

  // Update stats.
  if (stats_) {
    ++(stats_->num_allocs);
    stats_->bytes_in_use += num_bytes;
    if (stats_->bytes_in_use > stats_->peak_bytes_in_use) {
      VLOG(9) << "New Peak memory usage of " << stats_->bytes_in_use
              << " bytes.";
    }
    stats_->peak_bytes_in_use =
        std::max(stats_->peak_bytes_in_use, stats_->bytes_in_use);
    stats_->largest_alloc_size =
        std::max<std::size_t>(stats_->largest_alloc_size, num_bytes);
    bool ptr_inserted = size_map_.emplace(ptr, num_bytes).second;
    DCHECK(ptr_inserted);
  }
  VLOG(10) << Name() << " Allocated " << num_bytes << " at " << ptr;
  return ptr;
#else   // TF_CUDA_MALLOC_ASYNC_SUPPORTED
  return nullptr;
#endif  // TF_CUDA_MALLOC_ASYNC_SUPPORTED
}
void GpuMallocAsyncAllocator::DeallocateRaw(void* ptr) {
#if TF_GPU_MALLOC_ASYNC_SUPPORTED
  if (ptr == nullptr) return;
  // The lock is only needed when stats are enabled, but it must be around
  // the cuMemFreeAsync call as well to ensure consistency of the stats update.
  std::unique_lock<tsl::mutex> lock(lock_, std::defer_lock);
  if (stats_) {
    lock.lock();
  }
  if (auto result = GpuDriver::MemFreeAsync(reinterpret_cast<const CUdeviceptr&>(ptr),
                                   gpu_stream_)) {
    if (result == CUDA_ERROR_DEINITIALIZED) {
      // It happens with multi-GPU that TF free the GPU allocation after
      // the driver is unloaded. It is safe to ignore this error here.
      // TODO: Find how to fix the shutdown steps in TF.
      VLOG(1) << "Ignoring CUDA error: " << GetCudaErrorMessage(result);
    } else {
      size_t free, total;
      cuda::ScopedActivateExecutorContext scoped_activation{stream_exec_};
      cuMemGetInfo(&free, &total);
      LOG(ERROR) << "cudaFreeAsync failed to free " << ptr << ": "
                 << GetCudaErrorMessage(result)
                 << "\n Free memory/Total memory: " << free << "/" << total;
      if (stats_) {
        LOG(ERROR) << "Stats: " << stats_->DebugString();
      }
    }
  }

  // Updates the stats.
  if (stats_) {
    DCHECK(size_map_.contains(ptr));
    size_t size = size_map_[ptr];
    stats_->bytes_in_use -= size;
    size_map_.erase(ptr);
  }

  VLOG(10) << Name() << " Freed ptr: " << ptr;
#endif  // TF_GPU_MALLOC_ASYNC_SUPPORTED
}

bool GpuMallocAsyncAllocator::TracksAllocationSizes() const {
  return static_cast<bool>(stats_);
}

size_t GpuMallocAsyncAllocator::RequestedSize(const void* ptr) const {
  if (!stats_ || !ptr) return 0;
  tsl::mutex_lock l(lock_);
  return size_map_.at(ptr);
}

size_t GpuMallocAsyncAllocator::AllocatedSize(const void* ptr) const {
  if (!stats_ || !ptr) return 0;
  tsl::mutex_lock l(lock_);
  return size_map_.at(ptr);
}

std::optional<tsl::AllocatorStats> GpuMallocAsyncAllocator::GetStats() {
  if (!stats_) return std::nullopt;
  tsl::mutex_lock l(lock_);
  return *stats_;
}

bool GpuMallocAsyncAllocator::ClearStats() {
  if (!stats_) return false;
  tsl::mutex_lock l(lock_);
  stats_->num_allocs = 0;
  stats_->peak_bytes_in_use = stats_->bytes_in_use;
  stats_->largest_alloc_size = 0;
  return true;
}

void GpuMallocAsyncAllocator::SetStreamAndPreallocateMemory(void* stream) {
#if TF_GPU_MALLOC_ASYNC_SUPPORTED
  auto new_gpu_stream = static_cast<GpuStreamHandle>(stream);
  // We don't need to re-set the CUDA stream if this is the same stream
  if (gpu_stream_ != nullptr && new_gpu_stream != gpu_stream_) {
    LOG(FATAL) <<  // Crash OK.
        "Trying to set the stream twice. This isn't supported. ";
  }

  uint64_t pool_size_64 = 0;
  if (auto status = GpuDriver::GpuMemPoolGetAttribute(
          pool_, GpuDriver::MemPoolAttribute::kReleaseThreshold, &pool_size_64)) {
    LOG(FATAL) <<  // Crash OK.
        "Failed to get GPU memory pool attribute: " << status.message();
  }
  gpu_stream_ = new_gpu_stream;
  int64_t prealloc_size = 0;
  // TF_GPU_MALLOC_ASYNC_SUPPORTED_PREALLOC=-1 is a special value that
  // preallocates the total pool size.
  TF_CHECK_OK(tsl::ReadInt64FromEnvVar(
      "TF_GPU_MALLOC_ASYNC_SUPPORTED_PREALLOC", 0, &prealloc_size));
  if (prealloc_size == -1) {
    prealloc_size = pool_size_64;
  } else if (reserve_memory_) {
    prealloc_size = pool_size_64;
  }

  if (prealloc_size != 0) {
    void* ptr = AllocateRaw(0, prealloc_size);
    DeallocateRaw(ptr);
    VLOG(2) << Name() << " GpuMallocAsyncAllocator reserved the pool for "
            << prealloc_size << " bytes"
            << ". First ptr: " << ptr;
    ClearStats();
  }
#endif
}

}  // namespace stream_executor

/* Copyright 2023 The OpenXLA Authors.

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

// The ROCM-specific Driver library support, implementing the general Driver
// interface.

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_DRIVER_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_DRIVER_H_

#include "absl/container/node_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/gpu/gpu_driver.h"

namespace stream_executor {
namespace gpu {
// Formats hipError_t to output prettified values into a log stream.
// Error summaries taken from:
string ToString(hipError_t result);

// GpuContext wraps the device_ordinal.
class GpuContext {
 public:
  static GpuContext* FromOrdinal(int ordinal) {
     auto* ctxt = reinterpret_cast<GpuContext*>(static_cast<intptr_t>(ordinal) + 1);
     CHECK(ctxt != nullptr);
     return ctxt;
  }

  int device_ordinal() const {
    return static_cast<int>(reinterpret_cast<intptr_t>(this) - 1);
  }

  // Disallow copying and moving.
  GpuContext() = delete;
  GpuContext(GpuContext&&) = delete;
  GpuContext(const GpuContext&) = delete;
  GpuContext& operator=(GpuContext&&) = delete;
  GpuContext& operator=(const GpuContext&) = delete;
};

}  // namespace gpu

namespace rocm {

using MemorySpace = gpu::MemorySpace;
using ScopedActivateContext = gpu::ScopedActivateContext;

// TODO: this function shall be added to the GpuDriver API as well
absl::Status OccupancyGetMaxPotentialBlockSize(int* gridSize, int* blockSize,
                                               hipFunction_t func,
                                               size_t dynSharedMemPerBlk,
                                               int blockSizeLimit);

// Returns the current context set in ROCm. This is done by calling ROCm
// driver (e.g., this value is not our cached view of the current context).
hipCtx_t CurrentContextOrDie();
}  // namespace rocm
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_DRIVER_H_

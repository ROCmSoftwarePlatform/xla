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

#include "xla/service/gpu/runtime/command_buffer_memset.h"

namespace xla::gpu {

namespace {
#if TENSORFLOW_USE_ROCM

__global__ void memset32_kernel(uint32_t n, uint32_t value, uint32_t* dst)
{
   uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
   if (i < n) dst[i] = value;
}

#endif
}  // namespace

void* command_buffer_memset32_kernel() {
#if TENSORFLOW_USE_ROCM
  return reinterpret_cast<void*>(&memset32_kernel);
#else
  return nullptr;
#endif
}

}  // namespace xla::gpu 

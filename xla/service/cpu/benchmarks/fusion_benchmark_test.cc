/* Copyright 2024 The OpenXLA Authors.

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

#include <cstdint>
#include <random>
#include <string_view>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/shape_util.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/test_benchmark.h"

namespace xla::cpu {

static void BM_FusionF32(benchmark::State& state) {
  int64_t d0 = state.range(0);

  std::string_view hlo = R"(
    HloModule fusion_f32_$d0

    ENTRY e {
      p0 = f32[1,2,1,$d0,256] parameter(0)
      p1 = f32[1,2,1,$d0,256] parameter(1)
      p2 = f32[] parameter(2)
      c1 = f32[] constant(1)
      bcast = f32[1,2,1,$d0,256] broadcast(p2), dimensions={}
      multiply = f32[1,2,1,$d0,256] multiply(bcast, p1)
      subtract = f32[] subtract(c1, p2)
      bcast1 = f32[1,2,1,$d0,256] broadcast(subtract), dimensions={}
      multiply1 = f32[1,2,1,$d0,256] multiply(p0, p0)
      multiply2 = f32[1,2,1,$d0,256] multiply(bcast1, multiply1)
      ROOT add = f32[1,2,1,$d0,256] add(multiply, multiply2)
    }
  )";

  std::minstd_rand0 engine;

  auto shape = ShapeUtil::MakeShape(F32, {1, 2, 1, d0, 256});
  auto scalar = ShapeUtil::MakeShape(F32, {});
  auto p0 = *LiteralUtil::CreateRandomLiteral<F32>(shape, &engine, 1.0f, 0.1f);
  auto p1 = *LiteralUtil::CreateRandomLiteral<F32>(shape, &engine, 1.0f, 0.1f);
  auto p2 = *LiteralUtil::CreateRandomLiteral<F32>(scalar, &engine, 1.0f, 0.1f);

  std::vector<const Literal*> args = {&p0, &p1, &p2};
  CHECK_OK(RunHloBenchmark(state, hlo, args, {{"$d0", absl::StrCat(d0)}}));
}

BENCHMARK(BM_FusionF32)
    ->MeasureProcessCPUTime()
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(8192)
    ->Arg(16384);

}  // namespace xla::cpu

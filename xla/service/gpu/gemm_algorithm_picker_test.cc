/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/service/gpu/gemm_algorithm_picker.h"

#include <string>
#include <fstream>
#include <sstream>

#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gemm_rewriter.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_utils.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"
#include "tsl/protobuf/dnn.pb.h"

namespace xla::gpu {
namespace {

namespace m = ::xla::match;

class GemmAlgorithmPickerTest : public HloTestBase,
                                public ::testing::WithParamInterface<bool> {
 public:
  GemmAlgorithmPickerTest() { AutotunerUtil::ClearAutotuneResults(); }

  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = HloTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_cublaslt(GetParam());
    debug_options.set_xla_gpu_enable_triton_gemm(false);
    return debug_options;
  }

  void SetUp() override {
    const auto& gpu_cc = backend()
                             .default_stream_executor()
                             ->GetDeviceDescription()
                             .gpu_compute_capability();

    if (auto* procm = std::get_if<se::RocmComputeCapability>(&gpu_cc)) {
      if (GetDebugOptionsForTest().xla_gpu_enable_cublaslt() &&
          !procm->has_hipblaslt()) {
        GTEST_SKIP() << "No gpublas-lt support on this architecture!";
      }
    }
  }
};

TEST_P(GemmAlgorithmPickerTest, SetAlgorithm) {
  auto comp = backend()
                  .default_stream_executor()
                  ->GetDeviceDescription()
                  .cuda_compute_capability();
  if (comp.IsAtLeast(se::CudaComputeCapability::AMPERE)) {
    GTEST_SKIP() << "Skipping this test for Ampere+ as it is supported and "
                    "recommended with "
                    "the Nvidia Volta+ GPUs.";
  }

  constexpr absl::string_view kHlo = R"(
HloModule module

ENTRY main {
  %arg0 = f32[100,100]{1,0} parameter(0)
  %arg1 = f32[100,100]{1,0} parameter(1)
  ROOT %dot = f32[100,100]{1,0} dot(arg0, arg1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  auto module_cfg = GetModuleConfigForTest();
  TF_ASSERT_OK_AND_ASSIGN(auto m,
                          ParseAndReturnVerifiedModule(kHlo, module_cfg));

  se::Platform* platform = PlatformUtil::GetDefaultPlatform().value();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors,
                          PlatformUtil::GetStreamExecutors(platform));
  ASSERT_GT(executors.size(), 0);
  se::StreamExecutor* stream_exec = executors[0];
  bool changed = false;
  TF_ASSERT_OK_AND_ASSIGN(
      changed, RunHloPass(GemmRewriter(stream_exec->GetDeviceDescription()
                                           .gpu_compute_capability()),
                          m.get()));
  changed = false;
  DebugOptions opts;
  AutotuneConfig cfg{DeviceConfig{stream_exec, nullptr}, opts};
  TF_ASSERT_OK_AND_ASSIGN(changed,
                          RunHloPass(GemmAlgorithmPicker(cfg), m.get()));
  ASSERT_TRUE(changed);

  AutotuneResults results;
  TF_ASSERT_OK(AutotunerUtil::SerializeAutotuneResults(&results));
  ASSERT_EQ(results.results_size(), 1);
  auto& result = *results.mutable_results(0)->mutable_result();
  int64_t old_algo_id = result.algorithm().algo_id();
  int64_t new_algo_id = old_algo_id + 1;
  result.mutable_gemm()->set_algorithm(new_algo_id);

  AutotunerUtil::ClearAutotuneResults();
  TF_ASSERT_OK(AutotunerUtil::LoadAutotuneResults(results));

  // Now send the same module through GemmAlgorithmPicker again.  The dot should
  // have the new algorithm.
  TF_ASSERT_OK_AND_ASSIGN(m, ParseAndReturnVerifiedModule(kHlo, module_cfg));
  changed = false;
  TF_ASSERT_OK_AND_ASSIGN(
      changed, RunHloPass(GemmRewriter(stream_exec->GetDeviceDescription()
                                           .gpu_compute_capability()),
                          m.get()));
  changed = false;
  TF_ASSERT_OK_AND_ASSIGN(changed,
                          RunHloPass(GemmAlgorithmPicker(cfg), m.get()));
  ASSERT_TRUE(changed);

  SCOPED_TRACE(m->ToString());
  HloInstruction* dot;
  if (module_cfg.debug_options().xla_gpu_enable_cublaslt()) {
    ASSERT_THAT(m->entry_computation()->root_instruction(),
                GmockMatch(m::CustomCall(&dot)));
  } else {
    ASSERT_THAT(m->entry_computation()->root_instruction(),
                GmockMatch(m::GetTupleElement(m::CustomCall(&dot), 0)));
  }

  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          dot->backend_config<GpuBackendConfig>());
  const GemmBackendConfig& config = gpu_config.gemm_backend_config();
  EXPECT_EQ(config.selected_algorithm(), new_algo_id);
}

TEST_P(GemmAlgorithmPickerTest, GetAlgorithmWithoutDevice) {
  auto comp = backend()
                  .default_stream_executor()
                  ->GetDeviceDescription()
                  .cuda_compute_capability();
  if (comp.IsAtLeast(se::CudaComputeCapability::AMPERE)) {
    GTEST_SKIP() << "Skipping this test for Ampere+ as it is supported and "
                    "recommended with "
                    "the Nvidia Volta+ GPUs.";
  }

  constexpr absl::string_view kHlo = R"(
HloModule module

ENTRY main {
  %arg0 = f32[100,100]{1,0} parameter(0)
  %arg1 = f32[100,100]{1,0} parameter(1)
  ROOT %dot = f32[100,100]{1,0} dot(arg0, arg1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";
  TF_ASSERT_OK_AND_ASSIGN(
      auto m, ParseAndReturnVerifiedModule(kHlo, GetModuleConfigForTest()));

  se::Platform* platform = PlatformUtil::GetDefaultPlatform().value();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors,
                          PlatformUtil::GetStreamExecutors(platform));
  ASSERT_GT(executors.size(), 0);
  se::StreamExecutor* stream_exec = executors[0];

  bool changed = false;
  TF_ASSERT_OK_AND_ASSIGN(
      changed, RunHloPass(GemmRewriter(stream_exec->GetDeviceDescription()
                                           .gpu_compute_capability()),
                          m.get()));
  changed = false;

  DebugOptions opts;
  AutotuneConfig cfg{DeviceConfig{stream_exec, nullptr}, opts};

  TF_ASSERT_OK_AND_ASSIGN(changed,
                          RunHloPass(GemmAlgorithmPicker(cfg), m.get()));
  ASSERT_TRUE(changed);

  AutotuneResults results;
  TF_ASSERT_OK(AutotunerUtil::SerializeAutotuneResults(&results));
  ASSERT_EQ(results.results_size(), 1);
  auto& result = *results.mutable_results(0)->mutable_result();
  int64_t old_algo_id = result.algorithm().algo_id();
  int64_t new_algo_id = old_algo_id + 1;
  result.mutable_gemm()->set_algorithm(new_algo_id);

  AutotunerUtil::ClearAutotuneResults();
  TF_ASSERT_OK(AutotunerUtil::LoadAutotuneResults(results));

  auto module_cfg = GetModuleConfigForTest();
  // Now send the same module through GemmAlgorithmPicker again.  The dot should
  // have the new algorithm.
  TF_ASSERT_OK_AND_ASSIGN(m, ParseAndReturnVerifiedModule(kHlo, module_cfg));
  changed = false;

  DevicelessConfig deviceless_config{
      stream_exec->GetDeviceDescription().model_str(),
      stream_exec->GetDeviceDescription().cuda_compute_capability()};
  AutotuneConfig deviceless_cfg{deviceless_config, opts};
  TF_ASSERT_OK_AND_ASSIGN(
      changed, RunHloPass(GemmRewriter(stream_exec->GetDeviceDescription()
                                           .gpu_compute_capability()),
                          m.get()));
  changed = false;
  TF_ASSERT_OK_AND_ASSIGN(
      changed, RunHloPass(GemmAlgorithmPicker(deviceless_cfg), m.get()))
  ASSERT_TRUE(changed);

  SCOPED_TRACE(m->ToString());
  HloInstruction* dot;

  if (module_cfg.debug_options().xla_gpu_enable_cublaslt()) {
    ASSERT_THAT(m->entry_computation()->root_instruction(),
                GmockMatch(m::CustomCall(&dot)));
  } else {
    ASSERT_THAT(m->entry_computation()->root_instruction(),
                GmockMatch(m::GetTupleElement(m::CustomCall(&dot), 0)));
  }

  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          dot->backend_config<GpuBackendConfig>());
  const GemmBackendConfig& config = gpu_config.gemm_backend_config();

  EXPECT_EQ(config.selected_algorithm(), new_algo_id);
}


// Test that the alpha and beta fields of the GemmBackendConfig are updated.
// A bias must be present for the beta value to be set.
// In order to have a bias add fused, the bias term must be overwritable.
// We assume that we may not overwrite parameters of a computation. Hence, we
// use the third parameter to create a new value which can be overwritten and
// will be used as the bias. This negate(param_2) has no semantic use, it simply
// exists so that bias may be overwritten.
TEST_P(GemmAlgorithmPickerTest, RuntimeGemmSelection) {
  
  const char* hlo_text_non_zero = R"(
HloModule NonZeroAlphaBeta

ENTRY AddDotsFunc {
  x = f32[2500,300] parameter(0)
  y = f32[300,800] parameter(1)
  param_2 = f32[2500,800] parameter(2)
  bias = f32[2500,800] add(param_2,param_2)
  k = f32[] constant(3.0)
  k_broadcast = f32[2500, 800] broadcast(k), dimensions={}
  dot_a = f32[2500,800] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}, operand_precision={highest,highest}
  dot_a_multiplied = f32[2500, 800] multiply(dot_a, k_broadcast)
  ROOT out = f32[2500,800] add(dot_a_multiplied, bias)
}
)";

  const char* hlo_text_bias_epilogue = R"(
HloModule BiasEpilogue

ENTRY test {
  x = f32[2000,3000] parameter(0)
  y = f32[3000,450] parameter(1)
  z = f32[450] parameter(2)
  dot_a = f32[2000,450] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  z_bcast = f32[2000,450] broadcast(z), dimensions={1}
  ROOT out = f32[2000,450] add(dot_a, z_bcast)
}

)";

  const char* hlo_text_batched_bias_epilogue = R"(
HloModule test
    ENTRY test {
      x = f16[4,15,15] parameter(0)
      y = f16[15,31] parameter(1)
      dot_a = f16[4,15,31] dot(x, y), lhs_contracting_dims={2}, rhs_contracting_dims={0}
      b = f16[31] parameter(2)
      b_bcast = f16[4,15,31] broadcast(b), dimensions={2}
      ROOT out = f16[4,15,31] add(dot_a, b_bcast)
}
)";

  const char* hlo_text_relu_epilogue = R"(
HloModule ReluEpilogue

ENTRY test {
  x = f32[2,3] parameter(0)
  y = f32[3,4] parameter(1)
  dot_a = f32[2,4] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  c = f32[] constant(0)
  c_bcast = f32[2,4] broadcast(c), dimensions={}
  ROOT out = f32[2,4] maximum(dot_a, c_bcast)
}

)";

const char* hlo_text_relu_bias = R"(
HloModule test

ENTRY test {
  x = f32[2,3] parameter(0)
  y = f32[3,4] parameter(1)
  z = f32[4] parameter(2)
  dot_a = f32[2,4] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  z_bcast = f32[2,4] broadcast(z), dimensions={1}
  add = f32[2,4] add(dot_a, z_bcast)
  c = f32[] constant(0)
  c_bcast = f32[2,4] broadcast(c), dimensions={}
  ROOT out = f32[2,4] maximum(add, c_bcast)
}
)";

  const char* batched_relu_bias_hlo_text = R"(
HloModule test

ENTRY test {
  x = f32[2,3,4] parameter(0)
  y = f32[4,5,6] parameter(1)
  z = f32[3,5,6] parameter(2)
  dot_a = f32[2,3,5,6] dot(x, y), lhs_contracting_dims={2}, rhs_contracting_dims={0}, operand_precision={highest,highest}
  z_bcast = f32[2,3,5,6] broadcast(z), dimensions={1,2,3}
  add = f32[2,3,5,6] add(dot_a, z_bcast)
  c = f32[] constant(0)
  c_bcast = f32[2,3,5,6] broadcast(c), dimensions={}
  ROOT out = f32[2,3,5,6] maximum(add, c_bcast)
}
)";

  std::ifstream ifs("/tf/xla/input.hlo");
  if(!ifs)
    throw "Unable to open file";

  std::stringstream buffer;
  buffer << ifs.rdbuf();

  // if(!GetDebugOptionsForTest().xla_gpu_enable_cublaslt()) {
  //   GTEST_SKIP() << "This test must run with blas-lt enabled!";
  // }

  // DebugOptions debug_options = GetDebugOptionsForTest();
  // debug_options.set_xla_gpu_enable_cublaslt(GetParam());

  // TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
  //                         GetOptimizedModule(buffer.str()));

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(buffer.str()));

  TF_ASSERT_OK_AND_ASSIGN(auto dataflow, HloDataflowAnalysis::Run(*module));
  const auto params = module->entry_computation()->parameter_instructions();

  using DataType = half;
  std::vector<Literal> arguments(params.size());
  std::vector<Literal*> argument_ptrs(params.size());
  for (int i = 0; i < params.size(); ++i) {
    VLOG(0) << i << "th param shape = " << params[i]->shape();
#if 0
    arguments[i] = LiteralUtil::CreateFullWithDescendingLayout<DataType>
          (params[i]->shape().dimensions(), (DataType)(i+1));
#else
    TF_ASSERT_OK_AND_ASSIGN(arguments[i], xla::MakeFakeLiteral(
          params[i]->shape(), true, false));
#endif
    argument_ptrs[i] = &arguments[i];
  }
  //VLOG(0) << "Optimized HLO: " << module->ToString();

  //EXPECT_TRUE(RunAndCompare(buffer.str(), ErrorSpec{1e-5, 1e-5}));
  //  Actual: false (NOT_FOUND: Custom call target '__cublas$gemm' was not registered)
  //EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module), argument_ptrs, ErrorSpec{1e-5, 1e-5}));
  EXPECT_TRUE(RunAndCompare(std::move(module), argument_ptrs, ErrorSpec{1e-5, 1e-5}));
}

INSTANTIATE_TEST_SUITE_P(GemmAlgorithmPickerTestSuite, GemmAlgorithmPickerTest,
                         ::testing::Bool());

}  // namespace
}  // namespace xla::gpu

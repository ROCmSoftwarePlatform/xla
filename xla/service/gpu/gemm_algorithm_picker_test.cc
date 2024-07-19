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

#include <cstdint>
#include <variant>
#include <vector>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/autotuner_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gemm_rewriter.h"
#include "xla/service/gpu/variant_visitor.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor_pimpl.h"
#include "xla/tests/hlo_test_base.h"
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

  se::StreamExecutor *stream_exec() {
    return backend().default_stream_executor();
  }
  const se::DeviceDescription& gpu_device_desc() {
    return stream_exec()->GetDeviceDescription();
  }
  const se::GpuComputeCapability& gpu_comp() {
    return gpu_device_desc().gpu_compute_capability();
  }

  void SetUp() override {
    std::visit(VariantVisitor{
      [](const se::CudaComputeCapability& cc) {
        if(cc.IsAtLeastAmpere()) {
          GTEST_SKIP() << "Skipping this test for Ampere+ as it is supported and "
                    "recommended with "
                    "the Nvidia Volta+ GPUs.";
        }
      },
      [this](const se::RocmComputeCapability& cc) {
        if(GetDebugOptionsForTest().xla_gpu_enable_cublaslt() &&
          !cc.has_hipblaslt()) {
          GTEST_SKIP() << "No gpublas-lt support on this architecture!";
        }
      }},
      gpu_comp());
  }
};

TEST_P(GemmAlgorithmPickerTest, SkipAlgorithmsWithAccuracyCheck) {
  constexpr absl::string_view kHlo = R"(
HloModule module

ENTRY main {
  %arg0 = f32[100,100]{1,0} parameter(0)
  %arg1 = f32[100,100]{1,0} parameter(1)
  ROOT %dot = f32[100,100]{1,0} dot(arg0, arg1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  auto module_cfg = GetModuleConfigForTest();
  auto debug_opts = module_cfg.debug_options();
  size_t num_left1 = 0, num_left2 = 0;

TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHlo, module_cfg));

  {
    // Run first with default settings (autotune level = 4), keep the number of
    // algorithms left after autotuning
    TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloPass(
          GemmRewriter(gpu_comp()),
          module.get()));

    AutotuneConfig cfg{DeviceConfig{stream_exec(), nullptr}, debug_opts};
    GemmAlgorithmPicker gpicker(cfg);
    // Note that, we do not care if the algorithm index has been changed:
    // the thing matters is the # of algorithms left after sorting out.
    TF_ASSERT_OK_AND_ASSIGN(changed, RunHloPass(gpicker, module.get()));
    num_left1 = gpicker.num_algorithms_left();
    if(num_left1 < 2) {
      GTEST_SKIP() << "Too few algorithms left after the first step";
    }
  }

  // Clear cache before the second run!
  AutotunerUtil::ClearAutotuneResults();
  {
    // Run once again but now with autotune level 5 and embarassingly tight 
    // rtol which shall disqualify most of the algorithms.

    // Note that, we have "two sources of truth" for GemmAlgorithmPicker: i.e.,
    // debug_options are used to initialize both 'HloModuleConfig' and also 
    // 'AutotuneConfig'.
    debug_opts.set_xla_gpu_autotune_gemm_rtol(1e-12);
    debug_opts.set_xla_gpu_autotune_level(5);
    module->mutable_config().set_debug_options(debug_opts);
    TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloPass(
          GemmRewriter(gpu_comp()),
          module.get()));

    AutotuneConfig cfg{DeviceConfig{stream_exec(), nullptr}, debug_opts};
    GemmAlgorithmPicker gpicker(cfg);
    TF_ASSERT_OK_AND_ASSIGN(changed, RunHloPass(gpicker, module.get()));
    num_left2 = gpicker.num_algorithms_left();
  }
  // Assert that we have fewer algorithms left after the second run.
  ASSERT_TRUE(num_left1 > num_left2);
}

TEST_P(GemmAlgorithmPickerTest, SetAlgorithm) {
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

  bool changed = false;
  TF_ASSERT_OK_AND_ASSIGN(
      changed,
      RunHloPass(
          GemmRewriter(gpu_comp()),
          m.get()));
  changed = false;
  DebugOptions opts;
  AutotuneConfig cfg{DeviceConfig{stream_exec(), nullptr}, opts};
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
      changed,
      RunHloPass(
          GemmRewriter(gpu_comp()),
          m.get()));
  changed = false;
  TF_ASSERT_OK_AND_ASSIGN(changed,
                          RunHloPass(GemmAlgorithmPicker(cfg), m.get()));
  ASSERT_TRUE(changed);

  SCOPED_TRACE(m->ToString());
  HloInstruction* dot;
  ASSERT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::GetTupleElement(m::CustomCall(&dot), 0)));

  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          dot->backend_config<GpuBackendConfig>());
  const GemmBackendConfig& config = gpu_config.gemm_backend_config();
  EXPECT_EQ(config.selected_algorithm(), new_algo_id);
}

TEST_P(GemmAlgorithmPickerTest, GetAlgorithmWithoutDevice) {
  constexpr absl::string_view kHlo = R"(
HloModule module

ENTRY main {
  %arg0 = f32[100,100]{1,0} parameter(0)
  %arg1 = f32[100,100]{1,0} parameter(1)
  ROOT %dot = f32[100,100]{1,0} dot(arg0, arg1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";
  TF_ASSERT_OK_AND_ASSIGN(
      auto m, ParseAndReturnVerifiedModule(kHlo, GetModuleConfigForTest()));

  bool changed = false;
  TF_ASSERT_OK_AND_ASSIGN(
      changed,
      RunHloPass(
          GemmRewriter(gpu_comp()),
          m.get()));
  changed = false;

  DebugOptions opts;
  AutotuneConfig cfg{DeviceConfig{stream_exec(), nullptr}, opts};

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
      gpu_device_desc().model_str(), gpu_comp()};
  AutotuneConfig deviceless_cfg{deviceless_config, opts};
  TF_ASSERT_OK_AND_ASSIGN(
      changed,
      RunHloPass(
          GemmRewriter(gpu_comp()),
          m.get()));
  changed = false;
  TF_ASSERT_OK_AND_ASSIGN(
      changed, RunHloPass(GemmAlgorithmPicker(deviceless_cfg), m.get()))
  ASSERT_TRUE(changed);

  SCOPED_TRACE(m->ToString());
  HloInstruction* dot;

  ASSERT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::GetTupleElement(m::CustomCall(&dot), 0)));

  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          dot->backend_config<GpuBackendConfig>());
  const GemmBackendConfig& config = gpu_config.gemm_backend_config();

  EXPECT_EQ(config.selected_algorithm(), new_algo_id);
}

INSTANTIATE_TEST_SUITE_P(GemmAlgorithmPickerTestSuite, GemmAlgorithmPickerTest,
                         ::testing::Bool());

}  // namespace
}  // namespace xla::gpu

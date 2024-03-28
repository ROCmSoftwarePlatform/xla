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

#include <fstream>
#include <sstream>
#include "xla/error_spec.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_utils.h"

namespace xla {
namespace gpu {
namespace {

using SelectAndScatterTest = GpuCodegenTest;

TEST_F(SelectAndScatterTest, RegressionOOBWrites) {
  const char* hlo_text = R"(
HloModule TestModule

%select_op (a: f32[], b: f32[]) -> pred[] {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %compare = pred[] compare(f32[] %a, f32[] %b), direction=GE
}

%scatter_op (a: f32[], b: f32[]) -> f32[] {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %a, f32[] %b)
}

ge_F32.628 {
  lhs.629 = f32[] parameter(0)
  rhs.630 = f32[] parameter(1)
  ROOT compare.631 = pred[] compare(lhs.629, rhs.630), direction=GE
}

add_F32.632 {
  lhs.633 = f32[] parameter(0)
  rhs.634 = f32[] parameter(1)
  ROOT add.635 = f32[] add(lhs.633, rhs.634)
}

ENTRY %select_and_scatter (operand: f32[768,96,96,64]{3,2,1,0}, add.193: f32[768,48,48,64]{3,2,1,0}) -> f32[768,96,96,64]{3,2,1,0} {
  add.193 = f32[768,96,96,64]{3,2,1,0} parameter(1)
  add.620 = f32[768,48,48,64]{3,2,1,0} parameter(0)
  constant.627 = f32[] constant(0)
  ROOT %result = f32[768,96,96,64]{3,2,1,0} select-and-scatter(add.193, add.620, constant.627), window={size=1x3x3x1 stride=1x2x2x1 pad=0_0x0_1x0_1x0_0}, select=ge_F32.628, scatter=add_F32.632
}
)";
  EXPECT_TRUE(RunAndCompareNoHloPasses(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

template <class T>
std::vector<T*> MakePointerVector(std::vector<T>& input_vec) {
  std::vector<T*> output_pointers;
  output_pointers.reserve(input_vec.size());
  for (auto& input : input_vec) {
    output_pointers.push_back(&input);
  }
  return output_pointers;
}

void WriteLiteralToTempFile(const LiteralSlice& literal,
                            const std::string& name) {
  // Bazel likes for tests to write "debugging outputs" like these to
  // TEST_UNDECLARED_OUTPUTS_DIR.  This plays well with tools that inspect test
  // results, especially when they're run on remote machines.
  std::string outdir{tsl::testing::TmpDir()};

  auto* env = tsl::Env::Default();
  std::string filename = outdir + absl::StrFormat("/tempfile-%d-%s", env->NowMicros(), name);
  TF_CHECK_OK(tsl::WriteStringToFile(env, absl::StrCat(filename, ".txt"),
                                     literal.ToString()));
  LOG(ERROR) << "wrote Literal to " << name << " file: " << filename
             << ".txt";
}

tsl::StatusOr<xla::Literal> ReadLiteralFromProto(const std::string& name) {

  xla::LiteralProto proto;
  auto* env = tsl::Env::Default();
  TF_CHECK_OK(tsl::ReadBinaryProto(env, name, &proto));
  //*proto.mutable_shape() = ShapeUtil::MakeShape(F32, {42, 2}).ToProto();

  return xla::MutableLiteralBase::CreateFromProto(proto, true);
}

TEST_F(SelectAndScatterTest, SelectAndScatterPerformance) {

  std::ifstream ifs("/tf/xla/input.hlo");
  if(!ifs)
    throw "Unable to open file";

  std::stringstream buffer;
  buffer << ifs.rdbuf();

  HloModuleConfig config = GetModuleConfigForTest();

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(buffer.str(), 
          config));

  auto fake_arguments = xla::MakeFakeArguments(module.get(), 
        true, /*pseudo-random*/
        false /* use large range*/).value();
  auto arg_ptrs = MakePointerVector<xla::Literal>(fake_arguments);

  ErrorSpec error_spec{1e-2, 1e-2};
#if 1
  auto ref_module = module->Clone();  
  TF_ASSERT_OK_AND_ASSIGN(auto exec, CreateExecutable(std::move(module), true));

  auto& ref_runner = HloTestBase::reference_runner_;
  TF_ASSERT_OK_AND_ASSIGN(
      auto ref_exec, ref_runner.CreateExecutable(std::move(ref_module), true));

  TF_ASSERT_OK_AND_ASSIGN(auto truth, 
        ReadLiteralFromProto("/tf/xla/ROCM_results/expected.pb"));
  VLOG(0) << "Read expected literal from file.. running test";
  //TF_ASSERT_OK_AND_ASSIGN(auto truth, 
  //ref_runner.ExecuteWithExecutable(ref_exec.get(), arg_ptrs, nullptr));
  //WriteLiteralToTempFile(truth, "expected");

  for(int i = 0; i < 1; i++) {
    TF_ASSERT_OK_AND_ASSIGN(auto test_res, 
        HloTestBase::test_runner_.ExecuteWithExecutable(exec.get(), arg_ptrs, nullptr));
    if(i == 0) {
      //WriteLiteralToTempFile(test_res, "actual");
      EXPECT_TRUE(LiteralTestUtil::Near(truth, test_res, error_spec));
    }
  }
#else
  EXPECT_TRUE(RunAndCompare(std::move(module), 
      absl::Span< xla::Literal * const>(arg_ptrs.data(), arg_ptrs.size()), error_spec));
#endif
}

}  // namespace
}  // namespace gpu
}  // namespace xla

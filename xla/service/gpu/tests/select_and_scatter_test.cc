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

TEST_F(SelectAndScatterTest, SelectAndScatterPerformance) {

  std::ifstream ifs("/tf/xla/input.hlo");
  if(!ifs)
    throw "Unable to open file";

  std::stringstream buffer;
  buffer << ifs.rdbuf();

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(buffer.str()));
  auto fake_arguments = xla::MakeFakeArguments(module.get()).value();

  TF_ASSERT_OK_AND_ASSIGN(auto exec, CreateExecutable(std::move(module), true));

  for(int i = 0; i < 20; i++) {
     TF_ASSERT_OK_AND_ASSIGN(auto val, 
        HloTestBase::test_runner_.ExecuteWithExecutable(exec.get(), fake_arguments, nullptr));
  }
}

}  // namespace
}  // namespace gpu
}  // namespace xla

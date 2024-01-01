/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.1
==============================================================================*/

#include "xla/service/gpu/runtime/gpublas_lt_matmul.h"

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "xla/mlir/runtime/transforms/custom_call_encoding.h"
#include "xla/runtime/custom_call.h"
#include "xla/runtime/executable.h"
#include "xla/runtime/logical_result.h"
#include "xla/runtime/state.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/runtime/support.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/xla.pb.h"
#include "tsl/platform/status.h"

#if TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"
#endif

namespace xla {
#if GOOGLE_CUDA || TF_HIPBLASLT

using xla::runtime::CustomCall;
using xla::runtime::CustomCallAttrEncodingSet;
using xla::runtime::EnumAttrEncoding;
using xla::runtime::State;
using xla::runtime::StridedMemrefView;

namespace lmhlo_gpu = ::mlir::lmhlo_gpu;

//===----------------------------------------------------------------------===//
// Register cuBLASLt attributes decoding with the Xla runtime.
//===----------------------------------------------------------------------===//

namespace runtime {
XLA_RUNTIME_REGISTER_ENUM_ATTR_DECODING(se::gpu::BlasLt::Epilogue);
}  // namespace runtime

//===----------------------------------------------------------------------===//
// Encoding from MHLO attributes to Xla runtime enums.
//===----------------------------------------------------------------------===//

namespace gpu {

void PopulateCublasLtMatmulAttrEncoding(CustomCallAttrEncodingSet& encoding) {
  encoding.Add<EnumAttrEncoding<lmhlo_gpu::CublasLtMatmulEpilogueAttr,
                                lmhlo_gpu::CublasLtMatmulEpilogue,
                                se::gpu::BlasLt::Epilogue>>(
      [](lmhlo_gpu::CublasLtMatmulEpilogue value) -> se::gpu::BlasLt::Epilogue {
        return gpublas_lt::AsBlasLtEpilogue(value).value();
      });
}

//===----------------------------------------------------------------------===//
// cuBLASLt matmul custom call implementation.
//===----------------------------------------------------------------------===//

namespace {

absl::Status DoMatmul(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options, State<GemmConfig> gemm_config,
    State<MatmulPlanVec> matmul_plan, StridedMemrefView a,
    StridedMemrefView b, StridedMemrefView c, StridedMemrefView d,
    std::optional<StridedMemrefView> bias, std::optional<StridedMemrefView> aux,
    std::optional<StridedMemrefView> a_scale,
    std::optional<StridedMemrefView> b_scale,
    std::optional<StridedMemrefView> c_scale,
    std::optional<StridedMemrefView> d_scale,
    std::optional<StridedMemrefView> d_amax, int64_t algorithm,
    double alpha_real, double alpha_imag, double beta,
    DotDimensionNumbers dot_dims, se::gpu::BlasLt::Epilogue epilogue,
    absl::Span<const int32_t> precision) {
  se::Stream* stream = run_options->stream();

  // Find the gemm config for this instance of matmul.
  TF_ASSIGN_OR_RETURN(GemmConfig * config, gemm_config.GetOrCreate([&] {
    return ToAbsl(GetGemmConfig(
        a, b, d, algorithm, alpha_real, alpha_imag, beta, dot_dims.lhs_batch,
        dot_dims.lhs_contract, dot_dims.rhs_batch, dot_dims.rhs_contract,
        precision.empty() ? se::blas::kDefaultComputePrecision
                          : *absl::c_max_element(precision),
        c, bias));
  }));

  // Get the matmul plan for this instance of matmul:
  // by default we create it for 8 devices (which should be enough).
  TF_ASSIGN_OR_RETURN(auto planVec, matmul_plan.GetOrCreate([] {
    return MatmulPlanVec(8); 
  }));

  size_t devID = stream->parent()->device_ordinal();
  if(devID >= planVec->size()) {
    planVec->resize(devID + 1);
  }
  auto& plan = (*planVec)[devID];
  if(plan.get() == nullptr) {
    VLOG(2) << "Creating new plan for deviceID: " << devID;
    TF_ASSIGN_OR_RETURN(plan, se::gpu::BlasLt::GetMatmulPlan(stream, 
            *config, epilogue));
  } else {
    VLOG(2) << "Reusing plan for deviceID: " << devID;
  }

  TF_ASSIGN_OR_RETURN(auto algos, plan->GetAlgorithms());
  if (static_cast<size_t>(algorithm) >= algos.size()) {
    return absl::InternalError(
        absl::StrFormat("The requested gpublas-lt matmul "
                        "algorithm is not found. Total algorithms available: "
                        "%zu; requested: %zu",
                        algos.size(), static_cast<size_t>(algorithm)));
  }

  se::DeviceMemoryBase a_data = GetDeviceAddress(a);
  se::DeviceMemoryBase b_data = GetDeviceAddress(b);
  se::DeviceMemoryBase c_data = GetDeviceAddress(c);
  se::DeviceMemoryBase d_data = GetDeviceAddress(d);
  se::DeviceMemoryBase bias_data;
  if (bias.has_value()) bias_data = GetDeviceAddress(*bias);
  se::DeviceMemoryBase aux_data;
  if (aux.has_value()) aux_data = GetDeviceAddress(*aux);

  se::DeviceMemoryBase a_scale_data;
  if (a_scale.has_value()) a_scale_data = GetDeviceAddress(*a_scale);
  se::DeviceMemoryBase b_scale_data;
  if (b_scale.has_value()) b_scale_data = GetDeviceAddress(*b_scale);
  se::DeviceMemoryBase c_scale_data;
  if (c_scale.has_value()) c_scale_data = GetDeviceAddress(*c_scale);
  se::DeviceMemoryBase d_scale_data;
  if (d_scale.has_value()) d_scale_data = GetDeviceAddress(*d_scale);
  se::DeviceMemoryBase d_amax_data;
  if (d_amax.has_value()) d_amax_data = GetDeviceAddress(*d_amax);

  // if we can add BFC allocator here: since it also implements
  // DeviceMemoryAllocator interface: see GetStreamExecutorGpuDeviceAllocator
  se::OwningScratchAllocator<> scratch_allocator(
      stream->parent()->device_ordinal(), stream->parent()->BFCAllocatorHack);

  return plan->ExecuteOnStream(
      stream, a_data, b_data, c_data, d_data, bias_data, aux_data, a_scale_data,
      b_scale_data, c_scale_data, d_scale_data, d_amax_data, algos[algorithm],
      scratch_allocator);
}

}  // namespace

static absl::Status CublasLtMatmulImpl(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options, State<GemmConfig> gemm_config,
    State<MatmulPlanVec> matmul_plan, StridedMemrefView a,
    StridedMemrefView b, StridedMemrefView c, StridedMemrefView d,
    std::optional<StridedMemrefView> bias, std::optional<StridedMemrefView> aux,
    int64_t algorithm, double alpha_real, double alpha_imag, double beta,
    DotDimensionNumbers dot_dims, se::gpu::BlasLt::Epilogue epilogue,
    absl::Span<const int32_t> precision) {
  VLOG(3) << "Running CublasLtMatmul";
  std::optional<StridedMemrefView> a_scale, b_scale, c_scale, d_scale, d_amax;
  return DoMatmul(run_options, debug_options, gemm_config, matmul_plan, a, b, c,
                  d, bias, aux, a_scale, b_scale, c_scale, d_scale, d_amax,
                  algorithm, alpha_real, alpha_imag, beta, dot_dims, epilogue,
                  precision);
}

static absl::Status CublasLtMatmulF8Impl(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options, State<GemmConfig> gemm_config,
    State<MatmulPlanVec> matmul_plan, StridedMemrefView a,
    StridedMemrefView b, StridedMemrefView c, StridedMemrefView a_scale,
    StridedMemrefView b_scale, StridedMemrefView c_scale,
    StridedMemrefView d_scale, StridedMemrefView d,
    CustomCall::RemainingArgs remaining_args, int64_t algorithm,
    double alpha_real, double alpha_imag, double beta,
    DotDimensionNumbers dot_dims, se::gpu::BlasLt::Epilogue epilogue,
    absl::Span<const int32_t> precision) {
  VLOG(3) << "Running CublasLtMatmulF8";
  std::optional<StridedMemrefView> bias, d_amax, aux;
  int current_remaining_arg = 0;

  // Get bias, if present
  if (epilogue == se::gpu::BlasLt::Epilogue::kBias ||
      epilogue == se::gpu::BlasLt::Epilogue::kBiasThenReLU ||
      epilogue == se::gpu::BlasLt::Epilogue::kBiasThenGELU ||
      epilogue == se::gpu::BlasLt::Epilogue::kBiasThenGELUWithAux) {
    if (remaining_args.size() <= current_remaining_arg) {
      return absl::InternalError("Epilogue not present in CublasLtMatmulF8 op");
    }
    auto bias_or_failure =
        remaining_args.get<StridedMemrefView>(current_remaining_arg++);
    if (failed(bias_or_failure)) {
      return absl::InternalError("Failed to get epilogue");
    }
    bias = bias_or_failure.value();
  }

  // Get amax, if present
  if (remaining_args.size() > current_remaining_arg) {
    auto d_amax_or_failure =
        remaining_args.get<StridedMemrefView>(current_remaining_arg++);
    if (failed(d_amax_or_failure)) {
      return absl::InternalError("Failed to get d_amax");
    }
    d_amax = d_amax_or_failure.value();
  }

  return DoMatmul(run_options, debug_options, gemm_config, matmul_plan, a, b, c,
                  d, bias, aux, a_scale, b_scale, c_scale, d_scale, d_amax,
                  algorithm, alpha_real, alpha_imag, beta, dot_dims, epilogue,
                  precision);
}

//===----------------------------------------------------------------------===//
// cuBLASLt custom calls bindings and registration.
//===----------------------------------------------------------------------===//

template <typename... Ts>
auto BindMatmulAttributes(runtime::CustomCallBinding<Ts...> binding) {
  return std::move(binding)
      .template Attr<int64_t>("algorithm")
      .template Attr<double>("alpha_real")
      .template Attr<double>("alpha_imag")
      .template Attr<double>("beta")
      .template Attr<DotDimensionNumbers>("dot_dims")
      .template Attr<se::gpu::BlasLt::Epilogue>("epilogue")
      .template Attr<absl::Span<const int32_t>>("precision");
}

auto CublasLtMatmulCall(const char* name) {
  return CustomCall::Bind(name)
      .UserData<const ServiceExecutableRunOptions*>()
      .UserData<const DebugOptions*>()
      .State<GemmConfig>("uid")
      .State<MatmulPlanVec>("uid")
      .Arg<StridedMemrefView>()   // a
      .Arg<StridedMemrefView>()   // b
      .Arg<StridedMemrefView>()   // c
      .Arg<StridedMemrefView>();  // d
}

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    CublasLtMatmul, FunctionWrapper<CublasLtMatmulImpl>(), checks,
    BindMatmulAttributes(CublasLtMatmulCall("xla.gpu.cublas.lt.matmul")
                             .Value(std::optional<StridedMemrefView>())  // bias
                             .Value(std::optional<StridedMemrefView>())  // aux
                         ));

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    CublasLtMatmulBias, FunctionWrapper<CublasLtMatmulImpl>(), checks,
    BindMatmulAttributes(CublasLtMatmulCall("xla.gpu.cublas.lt.matmul.bias")
                             .Arg<StridedMemrefView>()                   // bias
                             .Value(std::optional<StridedMemrefView>())  // aux
                         ));

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    CublasLtMatmulAux, FunctionWrapper<CublasLtMatmulImpl>(), checks,
    BindMatmulAttributes(CublasLtMatmulCall("xla.gpu.cublas.lt.matmul.aux")
                             .Value(std::optional<StridedMemrefView>())  // bias
                             .Arg<StridedMemrefView>()                   // aux
                         ));

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    CublasLtMatmulBiasAux, FunctionWrapper<CublasLtMatmulImpl>(), checks,
    BindMatmulAttributes(CublasLtMatmulCall("xla.gpu.cublas.lt.matmul.bias.aux")
                             .Arg<StridedMemrefView>()  // bias
                             .Arg<StridedMemrefView>()  // aux
                         ));

auto CublasLtMatmulF8Call(const char* name) {
  return CustomCall::Bind(name)
      .UserData<const ServiceExecutableRunOptions*>()
      .UserData<const DebugOptions*>()
      .State<GemmConfig>("uid")
      .State<MatmulPlanVec>("uid")
      .Arg<StridedMemrefView>()   // a
      .Arg<StridedMemrefView>()   // b
      .Arg<StridedMemrefView>()   // c
      .Arg<StridedMemrefView>()   // a_scale
      .Arg<StridedMemrefView>()   // b_scale
      .Arg<StridedMemrefView>()   // c_scale
      .Arg<StridedMemrefView>()   // d_scale
      .Arg<StridedMemrefView>();  // d
}

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    CublasLtMatmulF8, FunctionWrapper<CublasLtMatmulF8Impl>(), checks,
    BindMatmulAttributes(
        CublasLtMatmulF8Call("xla.gpu.cublas.lt.matmul.f8").RemainingArgs()));

void RegisterMatmulCustomCalls(runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.gpu.cublas.lt.matmul", CublasLtMatmul);
  registry.Register("xla.gpu.cublas.lt.matmul.bias", CublasLtMatmulBias);
  registry.Register("xla.gpu.cublas.lt.matmul.aux", CublasLtMatmulAux);
  registry.Register("xla.gpu.cublas.lt.matmul.bias.aux", CublasLtMatmulBiasAux);
  registry.Register("xla.gpu.cublas.lt.matmul.f8", CublasLtMatmulF8);
}

}  // namespace gpu
#endif  // GOOGLE_CUDA || TF_HIPBLASLT
}  // namespace xla

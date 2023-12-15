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
#include <hip/hip_runtime.h>

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
    State<se::gpu::BlasLt::MatmulPlanPtr> matmul_plan, StridedMemrefView a,
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

  //int deviceId;
  //hipGetDevice(&deviceId);

  //auto blas_support = stream->parent()->AsBlas();

  //LOG(ERROR) << "DoMatmul deviceId:" << deviceId << " blas_support:" << blas_support;

  // Find the gemm config for this instance of matmul.
  TF_ASSIGN_OR_RETURN(GemmConfig * config, gemm_config.GetOrCreate([&] {
    return ToAbsl(GetGemmConfig(
        a, b, d, algorithm, alpha_real, alpha_imag, beta, dot_dims.lhs_batch,
        dot_dims.lhs_contract, dot_dims.rhs_batch, dot_dims.rhs_contract,
        precision.empty() ? se::blas::kDefaultComputePrecision
                          : *absl::c_max_element(precision),
        c, bias));
  }));

  // Get the matmul plan for this instance of matmul.
  TF_ASSIGN_OR_RETURN(auto plan, matmul_plan.GetOrCreate([&] {
    return ToAbsl(se::gpu::BlasLt::GetMatmulPlan(stream, *config, epilogue));
  }));

  TF_ASSIGN_OR_RETURN(auto algos, (*plan)->GetAlgorithms());

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

  se::OwningScratchAllocator<> scratch_allocator(
      stream->parent()->device_ordinal(), stream->parent()->GetAllocator());

  return (*plan)->ExecuteOnStream(
      stream, a_data, b_data, c_data, d_data, bias_data, aux_data, a_scale_data,
      b_scale_data, c_scale_data, d_scale_data, d_amax_data, algos[algorithm],
      scratch_allocator);
}

}  // namespace

se::blas::Transpose AsBlasTranspose(MatrixLayout::Order order) {
  // BLAS is column-major by default.
  return (order == MatrixLayout::Order::kColumnMajor)
             ? se::blas::Transpose::kNoTranspose
             : se::blas::Transpose::kTranspose;
}

static absl::Status CublasLtMatmulImpl(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options, State<GemmConfig> gemm_config,
    State<se::gpu::BlasLt::MatmulPlanPtr> matmul_plan, StridedMemrefView a,
    StridedMemrefView b, StridedMemrefView c, StridedMemrefView d,
    std::optional<StridedMemrefView> bias, std::optional<StridedMemrefView> aux,
    int64_t algorithm, double alpha_real, double alpha_imag, double beta,
    DotDimensionNumbers dot_dims, se::gpu::BlasLt::Epilogue epilogue,
    absl::Span<const int32_t> precision) {
  //VLOG(3) << "Running CublasLtMatmul";
  //std::optional<StridedMemrefView> a_scale, b_scale, c_scale, d_scale, d_amax;
  //return DoMatmul(run_options, debug_options, gemm_config, matmul_plan, a, b, c,
  //                d, bias, aux, a_scale, b_scale, c_scale, d_scale, d_amax,
  //                algorithm, alpha_real, alpha_imag, beta, dot_dims, epilogue,
  //                precision);

  se::DeviceMemoryBase lhs_data = GetDeviceAddress(a);
  se::DeviceMemoryBase rhs_data = GetDeviceAddress(b);
  se::DeviceMemoryBase output_data = GetDeviceAddress(d);
  const bool deterministic_ops = debug_options->xla_gpu_deterministic_ops();
  
  VLOG(3) << "Running GEMM";
  se::Stream* stream = run_options->stream();
  Shape output_shape = ToShape(d);
  auto blas_support = stream->parent()->AsBlas();

  
  TF_ASSIGN_OR_RETURN(GemmConfig * config, gemm_config.GetOrCreate([&] {
    return ToAbsl(GetGemmConfig(
        a, b, d, algorithm, alpha_real, alpha_imag, beta, dot_dims.lhs_batch,
        dot_dims.lhs_contract, dot_dims.rhs_batch, dot_dims.rhs_contract,
        precision.empty() ? se::blas::kDefaultComputePrecision
                          : *absl::c_max_element(precision), c, bias,
                          /*grad_x=*/false,/*grad_y=*/ false));
  }));
  
  auto lhs_layout = MatrixLayout{config->lhs_layout};
  auto rhs_layout = MatrixLayout{config->rhs_layout};
  auto output_layout = MatrixLayout{config->output_layout};
  
  int64_t m = output_layout.num_rows;
  int64_t n = output_layout.num_cols;
  int64_t k = lhs_layout.num_cols;
  
  se::blas::Transpose ta = AsBlasTranspose(lhs_layout.order);
  se::blas::Transpose tb = AsBlasTranspose(rhs_layout.order);
  
  //LOG(ERROR) << " ta:" << static_cast<int>(ta) << " tb:" << static_cast<int>(tb) << " m:" << m << " n:" << n << " k:" << k;
  
  std::string s_ta = "T";
  std::string s_tb = "N";
  
  if (output_layout.order != MatrixLayout::Order::kColumnMajor) {
      int64_t tmp = m;
      m = n;
      n = tmp;
  
      se::blas::Transpose tmp_trans = ta;
  
      if (tb == se::blas::Transpose::kTranspose) {
          ta = se::blas::Transpose::kNoTranspose;
      } else {
          ta = se::blas::Transpose::kTranspose;
      }
  
      if (tmp_trans== se::blas::Transpose::kTranspose) {
          tb = se::blas::Transpose::kNoTranspose;
      } else {
          tb = se::blas::Transpose::kTranspose;
      }
  }

  if (ta == se::blas::Transpose::kTranspose) {
      s_ta = "T";
  } else {
      s_ta = "N";
  }

  if (tb == se::blas::Transpose::kTranspose) {
      s_tb = "T";
  } else {
      s_tb = "N";
  }

  std::stringstream sstm;
  sstm << s_ta << "_" << s_tb << "_" << m << "_" << n << "_" << k; 

  int soltype = 1;
  int solidx = 0;

  blas_support->findsol(sstm.str(), soltype, solidx);
  
  if (soltype == 2) {
      //LOG(ERROR) << "Running rocblas";
      return RunGemm(*config, lhs_data, rhs_data, output_data, output_data,
                 deterministic_ops, stream, solidx);
  } else {
      //VLOG(3) << "Running CublasLtMatmul";
      //LOG(ERROR) << "Running CublasLtMatmul";
      std::optional<StridedMemrefView> a_scale, b_scale, c_scale, d_scale, d_amax;
      return DoMatmul(run_options, debug_options, gemm_config, matmul_plan, a, b, c,
                  d, bias, aux, a_scale, b_scale, c_scale, d_scale, d_amax,
                  algorithm, alpha_real, alpha_imag, beta, dot_dims, epilogue,
                  precision);
  }
}

static absl::Status CublasLtMatmulF8Impl(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options, State<GemmConfig> gemm_config,
    State<se::gpu::BlasLt::MatmulPlanPtr> matmul_plan, StridedMemrefView a,
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
      .State<se::gpu::BlasLt::MatmulPlanPtr>("uid")
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
      .State<se::gpu::BlasLt::MatmulPlanPtr>("uid")
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

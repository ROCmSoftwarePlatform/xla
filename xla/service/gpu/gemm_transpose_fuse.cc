/* Copyright 2019 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=
=============================================================================*/

#include "xla/service/gpu/gemm_transpose_fuse.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

namespace m = match;

template <typename Pattern>
auto OptionalBitcast(HloInstruction **optional_bitcast, Pattern pattern) {
  return m::AnyOf<HloInstruction>(m::Bitcast(optional_bitcast, pattern),
                                  std::move(pattern));
}

class GemmTransposeFuseVisitor : public DfsHloRewriteVisitor {
  se::GpuComputeCapability gpu_version_;

 public:
  explicit GemmTransposeFuseVisitor(const se::GpuComputeCapability& gpu_version)
      : gpu_version_(gpu_version) {}

  absl::Status HandleDot(HloInstruction *instr) override {
    if (!IsMatrixMultiplication(*instr)) {
      return absl::OkStatus();
    }
    return absl::OkStatus();
  }

  absl::StatusOr<bool> RewriteDotWithLeftTranspose(HloInstruction *gemm, 
        HloInstruction *opt_bitcast, HloInstruction *transpose) {

    GpuBackendConfig gpu_config =
      gemm->backend_config<GpuBackendConfig>().value();
    GemmBackendConfig& backend_config = *gpu_config.mutable_gemm_backend_config();

    const auto& dot_dims = backend_config.dot_dimension_numbers();
    const auto& lhs_contract = dot_dims.lhs_contracting_dimensions(),
              & rhs_contract = dot_dims.rhs_contracting_dimensions();
    // not sure if this is necessary, but who knows if we have several contracting dims ??
    if(!(lhs_contract.size() == 1 ||
         rhs_contract.size() == 1)) {
      VLOG(0) << "Contract dimensions do not match!";
      return false; 
    }
    
    auto old_lhs = opt_bitcast ? opt_bitcast : transpose;
    auto tgt_dim = old_lhs->shape().dimensions(lhs_contract[0]);
    //VLOG(0) << (opt_bitcast ? "bitcast LHS: " : "transpose LHS: ") <<  old_lhs->shape().ToString() << " tgt_dim: " << tgt_dim;

    // for bitcast => go to transpose and find a contiguous list of indices 
    // which correspond to that contracting dim..
    
    auto lhs = transpose->operands()[0];
    auto lhs_dims = lhs->shape().dimensions();
    int64_t i1 = 0, i2 = 0, prod = 1, s1 = -1, s2 = -1, i = 0;
    while(1) {
      if(prod >= tgt_dim) {
        while(i1 < i2 && lhs_dims[i1] == 1) i1++; // skip trivial dims
        if(prod == tgt_dim) { // [i1; i2) range
          if(s1 >= 0) {
            VLOG(0) << "Bitcast repeating dims!";
            return false;
          }
          s1 = i1, s2 = i2; // keep found range
        }
        if(i1 < i2) {
          prod /= lhs_dims[i1++]; // shift by one left
        }
      } else {
        if((uint64_t)i >= lhs_dims.size()) break;
        prod *= lhs_dims[i++];
        i2++;
      }
    }
    if(s1 < 0) {
      VLOG(0) << "Range not found!";
      return false;
    }
    // if(s1 + 1 != s2) {
    //   VLOG(0) << "Not handling complex gemms";
    //   return false;
    // } 

    //VLOG(0) << "range found: [" << s1 << ',' << s2 << ']';

    auto new_operands = gemm->operands();
    auto rhs_dims = new_operands[1]->shape().dimensions();
    std::vector< int64_t > out_dims, bcast_dims;
    out_dims.reserve(lhs_dims.size() + rhs_dims.size() - 2);
    bcast_dims.reserve(lhs_dims.size());

    int contract_dim = 0;
    for(uint32_t i = 0; i < lhs_dims.size(); i++) {
      if(i == s1) {
        contract_dim = out_dims.size();
        bcast_dims.push_back(tgt_dim);
        i = s2 - 1; // push the whole range whose product is a target_dim
      } else if(auto z = lhs_dims[i]; z != 1) {
        bcast_dims.push_back(z); // ignore trivial dims
        out_dims.push_back(z);
      }
    }
    // NOTE: we could immediately fill out_dims!!
    auto bitcast_shape = ShapeUtil::MakeShape(
              lhs->shape().element_type(), bcast_dims);
    // VLOG(0) << "Got bitcast shape: " << bitcast_shape <<
    //     " contract dim: " << contract_dim;

    if(bitcast_shape != lhs->shape()) {
      //VLOG(0) << "Bitcast of " << lhs->ToString() << " required!!!";
      lhs = lhs->AddInstruction(HloInstruction::CreateBitcast(
          bitcast_shape, lhs));
    }
    for(uint32_t i = 0; i < rhs_dims.size(); i++) {
      if(i != rhs_contract[0]) out_dims.push_back(rhs_dims[i]); 
    }
    // it could be that old gemm call has layout {0,1} and the new one {1,0}
    // no idea why it changed...
    auto layout_fixup = [](const auto& oldmm, Layout *newl) {
      auto& mm = *newl->mutable_minor_to_major();
      if(mm.size() == oldmm.size()) {
        mm.clear();
        for(auto s : oldmm) {
          mm.push_back(s);
        }
      }
    };

    auto el_type = gemm->shape().element_type();
    Shape out_shape;
    if(el_type == xla::TUPLE) { // need to wrap custom call into a tuple again..
      auto tuple_shapes = gemm->shape().tuple_shapes();
      auto oldL = tuple_shapes[0].layout();
      tuple_shapes[0] = ShapeUtil::MakeShape(tuple_shapes[0].element_type(), out_dims);
      layout_fixup(oldL.minor_to_major(), tuple_shapes[0].mutable_layout());

      out_shape = ShapeUtil::MakeTupleShape(tuple_shapes);
    } else {
      auto oldmm = gemm->shape().layout().minor_to_major();
      out_shape = ShapeUtil::MakeShape(el_type, out_dims);
      layout_fixup(oldmm, out_shape.mutable_layout());
    }
    new_operands[0] = lhs;

    // TODO: use CloneWithNewOperands()
    auto new_gemm = gemm->AddInstruction(HloInstruction::CreateCustomCall(
             out_shape, new_operands, gemm->custom_call_target()));
    if(new_gemm->operand_count() == 3) {
      xla::Cast<HloCustomCallInstruction>(new_gemm)
          ->set_output_to_operand_aliasing({{{}, {2, {}}}});
    }

    // if we have 3 dimensions and contracting dim is not zero => set it as a batch dim
    auto& dim_nums = *backend_config.mutable_dot_dimension_numbers();
    dim_nums.clear_lhs_contracting_dimensions();
    dim_nums.clear_lhs_batch_dimensions();
    dim_nums.add_lhs_contracting_dimensions(contract_dim);
    // TODO: figure out which dims do we need!!
    if(lhs_dims.size() == 3) {
      dim_nums.add_lhs_batch_dimensions(0);
    }

    int64_t lhs_batchdims_sz = dot_dims.lhs_batch_dimensions_size();
    int64_t lhs_stride = lhs_dims[lhs_batchdims_sz] *
                         lhs_dims[lhs_batchdims_sz + 1];
    int64_t rhs_stride = rhs_dims[lhs_batchdims_sz] *
                         rhs_dims[lhs_batchdims_sz + 1];

    backend_config.set_lhs_stride(lhs_stride);
    backend_config.set_rhs_stride(rhs_stride);
    TF_RETURN_IF_ERROR(new_gemm->set_backend_config(gpu_config));

    TF_RETURN_IF_ERROR(gemm->parent()->
                  ReplaceInstructionWithDifferentShape(gemm, new_gemm));
    return true;
  } 

  absl::Status HandleGetTupleElement(HloInstruction *instr) override 
  {
    HloInstruction *call;
    if (Match(instr,
          m::GetTupleElement(
          m::CustomCall(&call)))) {

      auto tgt = call->custom_call_target();
      if(!(tgt == kGemmCallTarget || tgt == kCublasLtMatmulCallTarget)) {
        return absl::OkStatus();
      }
        //VLOG(0) << "Match tuple element: " << call->ToString();
      auto out_shape = call->shape().tuple_shapes()[0];

      // check if we need to fix tuple's shape
      if(out_shape != instr->shape()) {
        //VLOG(0) << "Call: " << out_shape << " -- tuple: " << instr->shape();
        auto new_call = instr->AddInstruction(
          instr->CloneWithNewOperands(out_shape, instr->operands()));

        TF_RETURN_IF_ERROR(instr->parent()->
                  ReplaceInstructionWithDifferentShape(instr, new_call));

        //VLOG(0) << "Tuple replaced comp:\n" << new_call->parent()->ToString();
        this->MarkAsChanged();
      }
    } 
    return absl::OkStatus();
  }

  absl::Status HandleCustomCall(HloInstruction *instr) override 
  {
    auto tgt = instr->custom_call_target();
    if(!(tgt == kGemmCallTarget || tgt == kCublasLtMatmulCallTarget)) {
      return absl::OkStatus();
    }

    //VLOG(0) << "Handle CustomCall original comp:\n" << instr->parent()->ToString();
    // VLOG(0) << "op1: " << instr->operands()[0]->ToString();
    // VLOG(0) << "op2: " << instr->operands()[1]->ToString();

    HloInstruction *opt_bitcast = nullptr, 
        *transpose = nullptr, *op2;
    if (Match(instr,
              m::CustomCall(
                OptionalBitcast(&opt_bitcast, 
                    m::Transpose(&transpose).WithOneUser())
                    .WithOneUser(),
                m::Op(&op2)
              ))
        ) {
      auto comp = instr->parent();
      // VLOG(0) << "Got match:\n" << instr->ToString() 
      //       << " orig lhs: " << 
      //       (opt_bitcast ? opt_bitcast->ToString() : transpose->ToString());

      TF_ASSIGN_OR_RETURN(auto z, RewriteDotWithLeftTranspose(instr, 
            opt_bitcast, transpose));
      
      if(z) {
        //VLOG(0) << instr->ToString() << " HLO changed!!";
        this->MarkAsChanged(); // set HLO visitor changed flag
      }
    }
    return absl::OkStatus();
  }

}; // class GemmTransposeFuseVisitor

absl::StatusOr<bool> RunOnComputation(HloComputation *computation,
                               const se::GpuComputeCapability& gpu_version) {
  GemmTransposeFuseVisitor visitor(gpu_version);
  TF_RETURN_IF_ERROR(computation->Accept(&visitor));
  return visitor.changed();
}

}  // anonymous namespace


absl::StatusOr<bool> GemmTransposeFuser::Run(
    HloModule *module,
    const absl::flat_hash_set<absl::string_view> &execution_threads) {
  bool changed = false;
  // HACK HACK
  //return false;
  // HACK HACK
  for (HloComputation *computation :
       module->MakeNonfusionComputations(execution_threads)) {

     //VLOG(0) << "---------------------- original comp:\n" << computation->ToString();
    TF_ASSIGN_OR_RETURN(bool result,
                        RunOnComputation(computation, gpu_version_));
    if(result)
    //  VLOG(0) << "+++++++++++++++++++++++++ changed comp:\n" << computation->ToString();
    changed |= result;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla

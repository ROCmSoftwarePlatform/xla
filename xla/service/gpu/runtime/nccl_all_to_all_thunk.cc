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
==============================================================================*/

#include "xla/service/gpu/runtime/nccl_all_to_all_thunk.h"

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "mlir/IR/Value.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/nccl_api.h"
#include "xla/service/gpu/nccl_collective_thunk.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/service/gpu/qccl_library.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

using mlir::lmhlo_gpu::AllToAllStartOp;

#define CHKQCCL(cmd) \
  if(auto res = (cmd); res != QCCL_Result::OK) {   \
    return absl::InternalError(absl::StrFormat("%d: QCCL failed with %d", __LINE__, (int)res)); \
  }

namespace {

NcclAllToAllConfig GetNcclAllToAllConfig(AllToAllStartOp op) {
  NcclAllToAllConfig config;
  // FIXME(b/180174349): LMHLO AllToAll incorrectly has use_global_device_ids
  // attribute and it should be removed.
  config.config = GetNcclCollectiveConfigForMlir(op, std::nullopt);
  config.has_split_dimension = op.getSplitDimension().has_value();
  return config;
}

NcclAllToAllConfig GetNcclAllToAllConfig(const HloAllToAllInstruction* instr) {
  NcclAllToAllConfig config;
  // FIXME(b/180174349): LMHLO AllToAll incorrectly has use_global_device_ids
  // attribute and it should be removed.
  config.config = GetNcclCollectiveConfig(instr, std::nullopt);
  config.has_split_dimension = instr->split_dimension().has_value();
  return config;
}

}  // namespace

NcclAllToAllStartThunk::NcclAllToAllStartThunk(
    ThunkInfo thunk_info, NcclApi* nccl_api, AllToAllStartOp op,
    std::vector<NcclCollectiveThunk::Buffer> buffers,
    const DebugOptions& debug_options)
    : NcclCollectiveThunk(Thunk::kNcclAllToAllStart, thunk_info, nccl_api,
                          op.getIsSync()),
      config_(GetNcclAllToAllConfig(op)),
      buffers_(std::move(buffers)) {
  CHECK_EQ(config_.config.operand_count, buffers_.size());
  
  if(debug_options.xla_gpu_qccl_collectives() & 1) {
    if(auto status = SetupQCCL(); !status.ok()) {
      LOG(WARNING) << status.message();
    }
  }
}

NcclAllToAllStartThunk::NcclAllToAllStartThunk(
    ThunkInfo thunk_info, NcclApi* nccl_api,
    const HloAllToAllInstruction* instr,
    std::vector<NcclCollectiveThunk::Buffer> buffers,
    const DebugOptions& debug_options)
    : NcclCollectiveThunk(Thunk::kNcclAllToAllStart, thunk_info, nccl_api,
                          IsSyncCollective(instr)),
      config_(GetNcclAllToAllConfig(instr)),
      buffers_(std::move(buffers)) {
  CHECK_EQ(config_.config.operand_count, buffers_.size());

  if(debug_options.xla_gpu_qccl_collectives() & 1) {
    if(auto status = SetupQCCL(); !status.ok()) {
      LOG(WARNING) << status.message();
    }
  }
}

/*static*/ absl::Status NcclAllToAllStartThunk::CheckImplementable(
    AllToAllStartOp op, int64_t replica_count, int64_t partition_count) {
  auto status = [&]() -> absl::Status {
    std::optional<uint64_t> split_dim = op.getSplitDimension();
    for (mlir::Value operand : op.getInputs()) {
      TF_RETURN_IF_ERROR(IsValidOperand(operand, Thunk::kNcclAllToAll));
      Shape shape = GetShape(operand);
      if (split_dim &&
          !ShapeUtil::IsEffectivelyMostMajorDimension(shape, *split_dim)) {
        return absl::UnimplementedError(absl::Substitute(
            "all-to-all split dim $0 is not the most major in input shape $1",
            *split_dim, shape.ToString(/*print_layout=*/true)));
      }
    }
    return absl::OkStatus();
  };
  return AddOpDescription<NcclAllToAllStartThunk>(status(), op, replica_count,
                                                  partition_count);
}

/*static*/ absl::Status NcclAllToAllStartThunk::CheckImplementable(
    const HloAllToAllInstruction* instr, int64_t replica_count,
    int64_t partition_count) {
  auto status = [&instr]() -> absl::Status {
    std::optional<uint64_t> split_dim = instr->split_dimension();
    for (HloInstruction* operand : instr->operands()) {
      Shape shape = operand->shape();
      TF_RETURN_IF_ERROR(IsValidOperand(shape, Thunk::kNcclAllToAll));
      if (split_dim &&
          !ShapeUtil::IsEffectivelyMostMajorDimension(shape, *split_dim)) {
        return absl::UnimplementedError(absl::Substitute(
            "all-to-all split dim $0 is not the most major in input shape $1",
            *split_dim, shape.ToString(/*print_layout=*/true)));
      }
    }
    return absl::OkStatus();
  };
  return AddOpDescription<NcclAllToAllStartThunk>(
      status(), instr, replica_count, partition_count);
}

/*static*/ CollectiveOpGroupMode NcclAllToAllStartThunk::GetGroupMode(
    AllToAllStartOp op) {
  return GetNcclAllToAllConfig(op).config.group_mode;
}
/*static*/ CollectiveOpGroupMode NcclAllToAllStartThunk::GetGroupMode(
    const HloAllToAllInstruction* instr) {
  return GetNcclAllToAllConfig(instr).config.group_mode;
}

absl::Status NcclAllToAllStartThunk::RunNcclCollective(
    const ExecuteParams& params, se::Stream& stream,
    NcclApi::NcclCommHandle comm) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, buffers_,
                             config_.config.operand_element_type));

  if(IsQCCLAvailable()) {
    TF_ASSIGN_OR_RETURN(int32_t num_participants, nccl_api()->CommCount(comm));
    return RunQCCL(num_participants, config_.has_split_dimension, 
          device_buffers, stream);
  }

  return xla::gpu::RunAllToAll(nccl_api(), config_.has_split_dimension,
                               device_buffers, stream, comm);
}

bool NcclAllToAllStartThunk::IsQCCLAvailable() {
  return qccl_available_;
}

absl::Status NcclAllToAllStartThunk::SetupQCCL() {

  for(int i = 0; i < 8; i++) {
    CHKQCCL(qcclInit(i));
  }
  qccl_available_ = true;
  return absl::OkStatus();
}

static StatusOr< uint32_t > SizeInBytes(PrimitiveType element_type) {
  switch (element_type) {
    case S8:
    case F8E5M2:
    case F8E4M3FN:
      return 1;
    case PRED:
    case U8:
      return 1;
    case S32:
    case U32:
      return 4;
    case S64:
    case U64:
      return 8;
    case F16:
      return 2;
    case F32:
      return 4;
    case C64:
    case F64:
      return 8;
    case C128:
      return 16;
    case S16:
    case U16:
    case BF16:
      return 2;
    default:
      return absl::InternalError("Unknown datatype");
  }
}

absl::Status NcclAllToAllStartThunk::RunQCCL(int32_t num_participants, 
          bool has_split_dimension, std::vector<DeviceBufferPair>& buffers,
          se::Stream& stream) {

  uint32_t numSubscribedPeers = 1;
  int current_id = stream.parent()->device_ordinal();
  if (has_split_dimension) {

    for (DeviceBufferPair& buffer : buffers) {
      TF_RET_CHECK(buffer.element_count % num_participants == 0)
          << "Buffer was not an exact multiple of the number of participants.";

      auto sz = SizeInBytes(buffer.element_type).value();
      size_t chunk_elements = buffer.element_count / num_participants,
             chunk_sz = chunk_elements * sz;
            
      for (int peer = 0; peer < num_participants; ++peer) {
        se::DeviceMemoryBase send_slice =
            NcclApi::Slice(buffer.source_buffer, buffer.element_type,
                           peer * chunk_elements, chunk_elements);

        se::DeviceMemoryBase recv_slice =
            NcclApi::Slice(buffer.destination_buffer, buffer.element_type,
                           peer * chunk_elements, chunk_elements);

        auto inP = peer, outP = peer;
        CHKQCCL(qcclSendRecv(current_id, numSubscribedPeers, inP, recv_slice.opaque(), 
             chunk_sz, outP, send_slice.opaque(), chunk_sz));
      }
    }

  } else {
    TF_RET_CHECK(buffers.size() == num_participants)
        << "Number of inputs didn't match the number of participants.";

    VLOG(0) << current_id << " num_participants " << buffers.size();
    for (size_t i = 0; i < buffers.size(); ++i) {
      DeviceBufferPair& buffer = buffers[i];

      auto sz = SizeInBytes(buffer.element_type).value();
      size_t chunk_sz = buffer.element_count * sz;

      auto inP = i, outP = i;
      CHKQCCL(qcclSendRecv(current_id, numSubscribedPeers, inP, 
             buffer.destination_buffer.opaque(), 
             chunk_sz, outP, buffer.source_buffer.opaque(), chunk_sz));
    }
  }
  CHKQCCL(qcclRun(current_id, se::gpu::AsGpuStreamValue(&stream)));
  return absl::OkStatus();
}


absl::Status RunAllToAll(NcclApi* nccl_api, bool has_split_dimension,
                         std::vector<DeviceBufferPair>& buffers,
                         se::Stream& stream, NcclApi::NcclCommHandle comm) {
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing all-to-all from device ordinal: " << device_ordinal;

  TF_ASSIGN_OR_RETURN(int32_t num_participants, nccl_api->CommCount(comm));

  TF_RETURN_IF_ERROR(nccl_api->GroupStart());

  // AllToAll can operate in two modes. Either it specifies a split dimension,
  // in which case inputs are split and outputs concatenated in that dimension
  // (here, we only support dimension 0), or it takes a list of inputs
  // and produces a tuple of outputs.
  if (has_split_dimension) {

    VLOG(0) << device_ordinal << " num_participants " << num_participants
          << " -- num bufs " << buffers.size(); 
    for (DeviceBufferPair& buffer : buffers) {
      TF_RET_CHECK(buffer.element_count % num_participants == 0)
          << "Buffer was not an exact multiple of the number of participants.";

      size_t chunk_elements = buffer.element_count / num_participants;
      
      for (int peer = 0; peer < num_participants; ++peer) {
        se::DeviceMemoryBase send_slice =
            NcclApi::Slice(buffer.source_buffer, buffer.element_type,
                           peer * chunk_elements, chunk_elements);

        se::DeviceMemoryBase recv_slice =
            NcclApi::Slice(buffer.destination_buffer, buffer.element_type,
                           peer * chunk_elements, chunk_elements);

        TF_RETURN_IF_ERROR(nccl_api->Send(send_slice, buffer.element_type,
                                          chunk_elements, peer, comm, &stream));

        TF_RETURN_IF_ERROR(nccl_api->Recv(recv_slice, buffer.element_type,
                                          chunk_elements, peer, comm, &stream));
      }
    }
  } else {
    TF_RET_CHECK(buffers.size() == num_participants)
        << "Number of inputs didn't match the number of participants.";

    VLOG(0) << device_ordinal << " num_participants " << buffers.size();
    for (size_t i = 0; i < buffers.size(); ++i) {
      DeviceBufferPair& buffer = buffers[i];

      TF_RETURN_IF_ERROR(
          nccl_api->Send(buffer.source_buffer, buffer.element_type,
                         buffer.element_count, i, comm, &stream));

      TF_RETURN_IF_ERROR(
          nccl_api->Recv(buffer.destination_buffer, buffer.element_type,
                         buffer.element_count, i, comm, &stream));
    }
  }

  return nccl_api->GroupEnd();
}

}  // namespace gpu
}  // namespace xla

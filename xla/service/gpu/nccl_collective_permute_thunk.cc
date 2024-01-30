/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/nccl_collective_permute_thunk.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "xla/mlir_hlo/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/translate/mhlo_to_hlo/attribute_exporter.h"
#include "xla/xla_data.pb.h"

#if XLA_ENABLE_XCCL
#include "xla/stream_executor/gpu/gpu_stream.h"
#endif

namespace xla {
namespace gpu {

using mlir::lmhlo_gpu::CollectivePermuteStartOp;

namespace impl {

class PermuteAddressExchange {

  constexpr static size_t s_histo_size = 256;
  constexpr static size_t s_max_size_kb = 1024*1024; // max 1Gb mem

  struct Data {
    volatile uint32_t round = 0; // every call to this function increases internal round...
    se::DeviceMemoryBase target_addr; // whom we are going to send
  };

  tsl::Env *env_;
  absl::Mutex mtx_;
  // one for each ID (do we really need it ??) and one for each device_ordinal 
  absl::flat_hash_map< uint32_t, std::vector< Data > > map_;
  std::array< uint32_t, s_histo_size + 1 > size_histo_;

public:
  PermuteAddressExchange() { 
    env_ = tsl::Env::Default();
    size_histo_.fill(0);
  }

  ~PermuteAddressExchange() {
    for(uint32_t i = 0; i <= s_histo_size; i++) {
      if(size_histo_[i] != 0) {
        uint32_t left = i * s_max_size_kb / s_histo_size,
                 right = (i + 1) * s_max_size_kb / s_histo_size;
        VLOG(0) << "transfer-size{" << left << "," << right << "}Kb = " << size_histo_[i];
      }
    }
  }

  void update_histo(size_t nbytes) {
    // convert to Kb with round up
    nbytes = std::min((nbytes + 1023)/1024, s_max_size_kb);
    int bin = (double)nbytes / s_max_size_kb * s_histo_size;
    size_histo_[bin]++;
  }

  // returns target address of peerID (or blocks until it's available)
  se::DeviceMemoryBase wait_and_set(uint32_t OpID, int64_t myID, int64_t peerID, 
                                      const se::DeviceMemoryBase& my_recv_addr) {
    std::vector< Data > *pdata = nullptr;
    {
      absl::MutexLock _(&mtx_);
      pdata = &map_[OpID];
      int64_t maxID = std::max(myID, peerID);
      if(maxID >= pdata->size()) { // this should be executed only once..
        pdata->resize(std::max(maxID, (int64_t)8), {});
      }
    }
    (*pdata)[myID].target_addr = my_recv_addr;
    auto round = ++(*pdata)[myID].round; // increase round as identifier that we set the pointer
    if(peerID < 0) { // this indicates that peer is not given
      return se::DeviceMemoryBase{};
    }
    auto& peer = (*pdata)[peerID];
    while(peer.round < round) {
      //VLOG(0) << myID << " is waiting for peer: " << peerID << " on round: " << round;
      env_->SleepForMicroseconds(2);
    }
    //VLOG(0) << "GPU " << myID << " peer " << peerID << " is ready: " << peer.target_addr.opaque();
    return peer.target_addr;
  }
};

StatusOr< uint32_t > SizeInBytes(PrimitiveType element_type) {
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

CollectiveOpGroupMode GetGroupMode(CollectivePermuteStartOp op) {
  return GetCollectiveOpGroupMode(op.getChannelId().has_value(), std::nullopt)
      .value();
}


NcclP2PConfig GetNcclP2PConfig(CollectivePermuteStartOp op,
                               int64_t replica_count, int64_t partition_count) {
  NcclP2PConfig CPC;
  auto& config = CPC.config;

  config.operand_count = 1;
  const Shape shape = GetShape(op.getOperand());
  config.operand_element_type.push_back(shape.element_type());
  config.SetCollectiveOpKindAndID(op);
  config.group_mode = GetGroupMode(op);

  VLOG(2) << "Shape size: " << shape.ToString() << " " << ShapeUtil::ByteSizeOf(shape) 
      << " replica_cnt: " << replica_count << " partition_count: " << partition_count
      << " opID: " << config.op_id;

  // With a collective permute, all execution instances together form one
  // replica group.
  const int64_t num_participants =
      config.group_mode == CollectiveOpGroupMode::kCrossReplica
          ? replica_count
          : partition_count;
  config.replica_groups.emplace_back();
  ReplicaGroup& replica_group = config.replica_groups.front();
  for (int i = 0; i < num_participants; ++i) {
    replica_group.add_replica_ids(i);
  }

  const std::vector<std::pair<int64_t, int64_t>> source_target_pairs =
      ConvertNx2Attribute(op.getSourceTargetPairs()).value();

  for (const auto&[source,target]: source_target_pairs) {
    VLOG(2) << "permute: " << source << " -> " << target;
    CPC.id_to_source_target.insert({target, {}})
        .first->second.source = source;
    CPC.id_to_source_target.insert({source, {}})
        .first->second.target = target;
  }

  VLOG(2) << "Running GetNcclP2PConfig config!";
  return CPC;
}

// The collective permute is degenerate if all source-target pairs are identity,
// and all the IDs appear in the list.
bool IsDegenerate(CollectivePermuteStartOp op, int64_t replica_count,
                  int64_t partition_count) {
  const std::vector<std::pair<int64_t, int64_t>> source_target_pairs =
      ConvertNx2Attribute(op.getSourceTargetPairs()).value();
  // Each ID can appear only once as a source and as a target. So if all pairs
  // are identity, all IDs must appear in the list is the size == number of
  // replicas/partitions.
  const int64_t expected_size =
      op.getChannelId() ? partition_count : replica_count;
  return source_target_pairs.size() == expected_size &&
         absl::c_all_of(source_target_pairs,
                        [](const std::pair<int64_t, int64_t>& source_target) {
                          return source_target.first == source_target.second;
                        });
}

Status CheckImplementable(CollectivePermuteStartOp op) {
  TF_RETURN_IF_ERROR(NcclCollectiveThunk::CheckImplementable());
  return IsValidOperand(op.getOperand(), Thunk::kNcclCollectivePermute);
}

}  // namespace impl

static impl::PermuteAddressExchange s_addrExchange;

NcclCollectivePermuteStartThunk::NcclCollectivePermuteStartThunk(
    ThunkInfo thunk_info, CollectivePermuteStartOp op, int64_t replica_count,
    int64_t partition_count, const Buffer& buffer)
    : NcclCollectiveThunk(Thunk::kNcclCollectivePermuteStart, thunk_info,
                          op.getIsSync()),
      config_(GetNcclP2PConfig(op, replica_count, partition_count)),
      buffer_(buffer) { }

/*static*/ NcclP2PConfig NcclCollectivePermuteStartThunk::GetNcclP2PConfig(
    CollectivePermuteStartOp op, int64_t replica_count,
    int64_t partition_count) {
  return impl::GetNcclP2PConfig(op, replica_count, partition_count);
}

/*static*/ Status NcclCollectivePermuteStartThunk::CheckImplementable(
    CollectivePermuteStartOp op, int64_t replica_count,
    int64_t partition_count) {
  return AddOpDescription<NcclCollectivePermuteStartThunk>(
      impl::CheckImplementable(op), op, replica_count, partition_count);
}

/*static*/ bool NcclCollectivePermuteStartThunk::IsDegenerate(
    CollectivePermuteStartOp op, int64_t replica_count,
    int64_t partition_count) {
  return impl::IsDegenerate(op, replica_count, partition_count);
}

/*static*/ CollectiveOpGroupMode NcclCollectivePermuteStartThunk::GetGroupMode(
    CollectivePermuteStartOp op) {
  return impl::GetGroupMode(op);
}

Status NcclCollectivePermuteStartThunk::RunNcclCollective(
    const ExecuteParams& params, se::Stream& stream, ncclComm_t comm) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, {buffer_},
                             config_.config.operand_element_type));
  TF_RET_CHECK(device_buffers.size() == 1) << "Expected one buffer pair.";

  TF_ASSIGN_OR_RETURN(const GlobalDeviceId global_device_id,
                      params.nccl_params.GetGlobalDeviceId());
  TF_ASSIGN_OR_RETURN(
      const DeviceAssignment::LogicalID current_logical_id,
      params.nccl_params.device_assn->LogicalIdForDevice(global_device_id));
  const int64_t current_id =
      config_.config.group_mode == CollectiveOpGroupMode::kCrossReplica
          ? current_logical_id.replica_id
          : current_logical_id.computation_id;
  std::string device_string = GetDeviceString(params.nccl_params);

  const NcclP2PConfig::SourceTargetMapEntry source_target =
      NcclP2PConfig::GetSourceTarget(config_.id_to_source_target, current_id);

  int64_t use_memcpy_to_peer = 0;
  VLOG(2) << "Running collective NcclCollectivePermuteStartThunk";
  return ::xla::gpu::RunCollectivePermute(source_target, device_buffers[0],
                                          stream, comm, device_string,
                                          current_id, use_memcpy_to_peer);
}

#define CHKHIP(x) if(auto res = (x); res != hipSuccess) { \
  return tsl::errors::Internal("HIP call failed !"; \
}

Status RunCollectivePermute(NcclP2PConfig::SourceTargetMapEntry source_target,
                            DeviceBufferPair& buffer, se::Stream& stream,
                            ncclComm_t comm, absl::string_view device_string,
                            int64_t current_id, int64_t use_memcpy_to_peer) {
#if XLA_ENABLE_XCCL
  // Determine the source and target IDs for this instance. The source ID is the
  // ID which will copy its data to this instance. The destination ID is the ID
  // to which this instance will copy its data. Either are optional.
  //
  // No source and no dest:
  //  - this instance does not actually participate, no one send it any data and
  //    it does not have to send any data as well. Since there is no dest,
  //    just memzero() the dest buffer as required by the collective permute
  //    semantics.
  //
  // No source, dest present:
  //  - This instance has to send data to 'dest' Issue an send of the input.
  //    Since there is no source, memzero the dest buffer.
  //
  // Source present, no destination:
  //  - This instance received data from the source, does not have to send data
  //    to anyone, Issue a receive.
  //
  // Source and dest both present:
  //   - Issue a send of the input to dest, receive for the output from the
  //     src.
  //
  //

  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(2) << "Performing collective permute from device ordinal: "
          << device_ordinal;

  const std::optional<int64_t> source_id = source_target.source;
  const std::optional<int64_t> target_id = source_target.target;

  se::DeviceMemoryBase src_addr = buffer.source_buffer;
  se::DeviceMemoryBase dest_addr = buffer.destination_buffer;

  VLOG(2) << absl::StreamFormat("%s : id = %d, source_id = %d, target_id = %d",
                                device_string, current_id,
                                source_id.value_or(-1), target_id.value_or(-1));

  se::gpu::GpuStreamHandle gpu_stream = se::gpu::AsGpuStreamValue(&stream);

  // ncclGroupStart/end API is needed only if we will issue both ncclSend and
  // ncclRecv API calls.
  const bool is_nccl_group_needed = (target_id && source_id);
  if(use_memcpy_to_peer == 0) {
    if (is_nccl_group_needed) {
      XLA_CUDA_RETURN_IF_ERROR(ncclGroupStart());
    }
    TF_ASSIGN_OR_RETURN(auto dtype_and_multiplier,
                      ToNcclDataTypeAndCountMultiplier(
                          buffer.element_type, Thunk::kNcclCollectivePermute));
    ncclDataType_t dtype = dtype_and_multiplier.first;
    int64_t element_count = buffer.element_count * dtype_and_multiplier.second;

    // Send source buffer to target peer if needed.
    if (target_id) {
      VLOG(2) << absl::StreamFormat(
          "%s : Calling ncclSend(sendbuff=%p, count=%d, peer=%d "
          "comm=%p, stream=%p)",
          device_string, src_addr.opaque(), element_count, *target_id,
          static_cast<const void*>(comm), gpu_stream);
      XLA_CUDA_RETURN_IF_ERROR(ncclSend(src_addr.opaque(), element_count, dtype,
                                      *target_id, comm, gpu_stream));
    }

    // Receive data from the source peer to the destination buffer.
    if (source_id) {
      VLOG(2) << absl::StreamFormat(
        "%s : Calling ncclRecv(recvbuff=%p, count=%d, peer=%d comm=%p, "
        "stream=%p)",
        device_string, dest_addr.opaque(), element_count, *source_id,
        static_cast<const void*>(comm), gpu_stream);
      XLA_CUDA_RETURN_IF_ERROR(ncclRecv(dest_addr.opaque(), element_count, dtype,
                                      *source_id, comm, gpu_stream));
    }
    if (is_nccl_group_needed) {
      XLA_CUDA_RETURN_IF_ERROR(ncclGroupEnd());
    }
  } else { // use_memcpy_to_peer
    auto target_addr = s_addrExchange.wait_and_set(111, current_id, target_id.value_or(-1), dest_addr);
    
    if(target_addr != nullptr) { // this indicates that target_id is given
      TF_ASSIGN_OR_RETURN(auto nbytes, impl::SizeInBytes(buffer.element_type));
      nbytes *= buffer.element_count;
      //VLOG(0) << "Bytes to be sent: " << nbytes;
      stream.ThenMemcpyD2D(&target_addr, src_addr, nbytes);
      //  CHKHIP(hipMemcpyPeerAsync(target_addr,
      //         *target_id, src_addr.opaque(), current_id, size, gpu_stream));
      if(device_ordinal == 0) {
        s_addrExchange.update_histo(nbytes); // only one GPU saves stats
      }
    }
  } // use_memcpy_to_peer

  if (!source_id) {
    // If there is no source peer, i.e. no one send us any data, zero out dest
    // buffer.
    VLOG(2) << absl::StreamFormat("%s : collective-Permute: Issuing MemZero",
                                  device_string);
    stream.ThenMemZero(&dest_addr, dest_addr.size());
  }
  
  return OkStatus();
#else   // XLA_ENABLE_XCCL
  return Unimplemented(
      "NCCL support is not available: this binary was not built with a CUDA "
      "compiler, which is necessary to build the NCCL source library.");
#endif  // XLA_ENABLE_XCCL
}

}  // namespace gpu
}  // namespace xla

/* Copyright 2021 The OpenXLA Authors.

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
#include "xla/service/gpu/qccl_library.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/mlir_hlo/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "xla/service/gpu/nccl_api.h"
#include "xla/service/gpu/nccl_collective_thunk.h"
#include "xla/service/gpu/thunk.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/stream.h"
#include "tsl/platform/errors.h"

namespace xla::gpu {

#define CHKQCCL(cmd) \
  if(auto res = (cmd); res != QCCL_Result::OK) {   \
    return absl::InternalError(absl::StrFormat("%d: QCCL failed with %d", __LINE__, (int)res)); \
  }

template < class T >
struct Matrix : std::vector< T > {

  using Base = std::vector< T >;

  Matrix(uint32_t nrows, uint32_t ncols, const T& val = {}) 
            : Base(ncols*nrows, val),
        m_nrows(nrows), m_ncols(ncols) {
  }
  // access the ith row
  T *operator[](uint32_t i) { 
    return Base::data() + i*m_ncols;
  }
  const T *operator[](uint32_t i) const {
    return Base::data() + i*m_ncols;
  }
  auto numRows() const {
    return m_nrows;
  }
  auto numCols() const {
    return m_ncols;
  }

  std::string printRow(uint32_t row) const {
    auto prow = (*this)[row];
    std::ostringstream oss;
    oss << '{';
    for(uint32_t i = 0; i < m_ncols; i++) {
      oss << prow[i];
      if(i < m_ncols-1) oss << ',';
    }
    oss << '}';
    return oss.str();
  }

private:
  uint32_t m_nrows, m_ncols;
};

struct Node {
  union {
    uint32_t in;    // this node receives form Node[in]
    uint32_t ofs;   // chunk offset for traffic splitting
  };
  union {
    uint32_t out;   // this Node sends to Node[out]
    uint32_t size;  // chunk size for traffic splitting
  };
};

struct NodeMatrix : Matrix< Node > {

  NodeMatrix(uint32_t nrows, uint32_t ncols, const Node& val = {}) :
      Matrix< Node >(nrows, ncols, val) {}
};

std::ostream& operator<<(std::ostream& ofs, const Node& n) {
  return ofs << '(' << n.in << ',' << n.out << ')';
}

using mlir::lmhlo_gpu::CollectivePermuteStartOp;

NcclCollectivePermuteStartThunk::NcclCollectivePermuteStartThunk(
    ThunkInfo thunk_info, NcclApi* nccl_api,
    const HloCollectivePermuteInstruction* instr, int64_t replica_count,
    int64_t partition_count, const Buffer& buffer,
    const DebugOptions& debug_options)
    : NcclCollectiveThunk(Thunk::kNcclCollectivePermuteStart, thunk_info,
                          nccl_api, IsSyncCollective(instr)),
      config_(GetNcclP2PConfig(instr, replica_count, partition_count)),
      buffer_(buffer) {
  if(debug_options.xla_gpu_qccl_collectives() & 2) {
    if(auto status = SetupQCCL(); !status.ok()) {
      LOG(WARNING) << status.message();
    }
  }
}

NcclCollectivePermuteStartThunk::~NcclCollectivePermuteStartThunk() = default;

bool NcclCollectivePermuteStartThunk::IsQCCLAvailable() {
  return (bool)commGraph_;
}

absl::Status NcclCollectivePermuteStartThunk::SetupQCCL() {

  constexpr static uint32_t s_bogus = 0xFFFFFFFFu; // to catch uninitialized entries
  size_t nNodes = config_.id_to_source_target.size();

  static const std::vector< std::pair< uint32_t, double >> s_peersMap = {
    { 0, 1.0 }, // 1
    { 0, 1.0 }, // 2
    { 0, 1.0 }, // 3
    { 1, 0.7 }, // 4
    { 2, 0.5 }, // 5
    { 3, 0.4 }, // 6
    { 4, 0.35 }, // 7
    { 5, 0.33 }, // 8
  };
  if(nNodes - 1 >= s_peersMap.size()) {
    nExtraPeers_ = nNodes - 3;
    splitFactor_ = 0.2; 
  } else {
    std::tie(nExtraPeers_, splitFactor_) = s_peersMap[nNodes - 1];
  }
  // disable hops..
  nExtraPeers_ = 0, splitFactor_ = 1.0;

  // we use first nNodes rows for communication graph and the last one to store
  // offsets / sizes
  auto commGraph = std::make_unique< NodeMatrix >(nNodes + 1, nExtraPeers_ + 1, 
                            Node{s_bogus, s_bogus});
  auto& graph = *commGraph;
  //VLOG(0) << " total replica groups: " << config_.id_to_source_target.size();

  for(const auto& [a,b]: config_.id_to_source_target) {
    if(b.target) graph[a][0].out = *b.target;  // gpu i sends to b.target
    if(b.source) graph[a][0].in = *b.source;   // gpu i receives from b.source
    // we have to preallocate mem for exchange buffers before..
    CHKQCCL(qcclInit(a));
  }
  // TODO: if some in/out is not set here, we still could use QCCL
  // but just use NULL buffers
  size_t elem_sz = ShapeUtil::ByteSizeOfPrimitiveType(
                                        config_.config.operand_element_type[0]);
  auto *extra = graph[nNodes];
  size_t size = elem_sz * buffer_.element_count;
  extra[0].ofs = 0;
  extra[0].size = (uint32_t)(size * splitFactor_) & ~15;

  // remaining is to be split evenly between nExtraPeers_
  size_t ofs = extra[0].size, remaining = size - ofs, 
         step = nExtraPeers_ > 0 ? (remaining / nExtraPeers_) & ~15 : 0;
  for(uint32_t i = 1; i <= nExtraPeers_; i++, ofs += step) {
    extra[i].ofs = ofs;
    extra[i].size = step;
  }
  extra[nExtraPeers_].size = size - extra[nExtraPeers_].ofs;

  // the # of incoming links and outgoing links (the target link is already counted)
  std::vector< uint32_t > numLinks(nNodes, 1);
  for(uint32_t i = 0; i < nNodes; i++) {

    auto t = graph[i][0].out; // target node for GPU i
    // iterate until all outgoing links for GPU i are filled
    //VLOG(0) << "Examining " << i << " -> " << t;
#if 1
    for(uint32_t jc = i + 1, n = 1; 
                         jc < 500 && n <= nExtraPeers_; jc++) {
      uint32_t j = jc % nNodes; // sometimes we need 2 rounds
      // skip self, the target node, and nodes with too many extra peers
      if(i == j || t == j || numLinks[j] > nExtraPeers_) { 
        continue;
      }
      // graph[j][z] can include (i,t) if:
      // 1. there is no (i,t) in graph[j] - in that row
      // 2. i != j and t != j 

      // NOTE: for N extra peers we need N rounds ??/
      // check if this slot is empty
      if(graph[j][n].in != s_bogus)
        continue;
      // increase the number of nodes processed use node j as a gateway to 
      // send data from i to t
      auto z = n++;
      numLinks[j]++;
      graph[j][z].in = i;  // node j receives z-th piece from node i
      graph[j][z].out = t; // node j forwards z-th piece to node t
    }
#else
    // alternative approach: use gateways living on target nodes
    // so that we have one direct write and one direct read kernel
    if(nExtraPeers_ != 1) {
      throw std::runtime_error("This approach works only for one peer!");
    }
    int z = 1;
    graph[t][z].in = i;   // node t receives z-th piece from node i
    graph[t][z].out = t;  // node t saves z-th piece to itself
#endif
  } // for i nNodes
  // VLOG(0) << "Legend: GPU x send: (i,j): gpu[x] receives from gpu[i] and sends to gpu[j]";
  // for(uint32_t i = 0; i < nNodes; i++) { 
  //   VLOG(0) << "GPU " << i << " send: " << graph.printRow(i);
  // }
  for(const auto& a : graph) {
    if(a.in == s_bogus || a.out == s_bogus) {
      return absl::InternalError("Unsupported permutation op: QCCL is disabled for this thunk!");
    }
  }
  // set communication graph which indicates that initalization was successful
  commGraph_.swap(commGraph); 
  return absl::OkStatus();
}

absl::Status NcclCollectivePermuteStartThunk::RunQCCL(DeviceBufferPair& buffer, 
        se::Stream& stream, int64_t current_id) {

  auto src_addr = buffer.source_buffer,
       dest_addr = buffer.destination_buffer;

  const auto& graph = *commGraph_;
  const auto *V = graph[current_id];
  // the last row is used to store "traffic splitting" data
  const auto *extra = graph[graph.numRows()-1];
  uint32_t numSubscribedPeers = 1 + nExtraPeers_;
  for(int i = 0; i <= nExtraPeers_; i++) {
      int inP = V[i].in, outP = V[i].out;
      auto size = extra[i].size;
      if(i == 0) {
        CHKQCCL(qcclSendRecv(current_id, numSubscribedPeers, inP, dest_addr.opaque(), 
            size, outP, src_addr.opaque(), size));
      } else {
        CHKQCCL(qcclGatewaySend(current_id, numSubscribedPeers, inP, outP, 
              extra[i].ofs, size));
      }
  }
  CHKQCCL(qcclRun(current_id, se::gpu::AsGpuStreamValue(&stream)));
  return absl::OkStatus();
}

} // namespace xla::gpu

/* Copyright 2024 The OpenXLA Authors. All Rights Reserved.

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

#if TENSORFLOW_USE_ROCM

#include <memory>
#include <utility>

#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xla/backends/profiler/gpu/rocm_collector.h"
#include "xla/backends/profiler/gpu/rocm_tracer.h"
#include "xla/tsl/util/env_var.h"
#include "tsl/platform/abi.h"
#include "tsl/platform/env_time.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/thread_annotations.h"
#include "tsl/profiler/backends/cpu/annotation_stack.h"
#include "tsl/profiler/lib/profiler_factory.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/utils/parse_annotation.h"
#include "tsl/profiler/utils/xplane_builder.h"
#include "tsl/profiler/utils/xplane_schema.h"
#include "tsl/profiler/utils/xplane_utils.h"

namespace xla {
namespace profiler {

using tensorflow::ProfileOptions;
using tsl::mutex;
using tsl::mutex_lock;
using tsl::Status;
using tsl::profiler::Annotation;
using tsl::profiler::AnnotationStack;
using tsl::profiler::FindOrAddMutablePlaneWithName;
using tsl::profiler::GetStatTypeStr;
using tsl::profiler::GpuPlaneName;
using tsl::profiler::kDeviceVendorAMD;
using tsl::profiler::kThreadIdOverhead;
using tsl::profiler::ParseAnnotationStack;
using tsl::profiler::ProfilerInterface;
using tsl::profiler::RegisterProfilerFactory;
using tsl::profiler::StatType;
using tsl::profiler::XEventBuilder;
using tsl::profiler::XEventMetadata;
using tsl::profiler::XLineBuilder;
using tsl::profiler::XPlaneBuilder;
using tsl::profiler::XSpace;

namespace se = ::stream_executor;

// GpuTracer for ROCm GPU.
class GpuTracer : public profiler::ProfilerInterface {
 public:
  GpuTracer(RocmTracer* rocmtracer) : rocm_tracer_(rocmtracer) {
    LOG(ERROR) << "GpuTrace with rocprofv3...\n";
    Start();
    LOG(INFO) << "GpuTracer created...";
  }
  ~GpuTracer() override {}

  // GpuTracer interface:
  absl::Status Start() override;
  absl::Status Stop() override;
  absl::Status CollectData(XSpace* space) override;

 private:
  absl::Status DoStart();
  absl::Status DoStop();

  RocmTracerOptions GetRocmTracerOptions();
  RocmTraceCollectorOptions GetRocmTraceCollectorOptions(uint32_t num_gpus);

  enum State {
    kNotStarted,
    kStartedOk,
    kStartedError,
    kStoppedOk,
    kStoppedError
  };
  State profiling_state_ = State::kNotStarted;

  RocmTracer* rocm_tracer_;
  std::unique_ptr<RocmTraceCollector> rocm_trace_collector_;
};

RocmTracerOptions GpuTracer::GetRocmTracerOptions() {
  // TODO(rocm-profiler): We need support for context similar to CUDA ?
  RocmTracerOptions options;
  return options;
}

RocmTraceCollectorOptions GpuTracer::GetRocmTraceCollectorOptions(
    uint32_t num_gpus) {
  RocmTraceCollectorOptions options;
  options.max_callback_api_events = 2 * 1024 * 1024;
  options.max_activity_api_events = 2 * 1024 * 1024;
  options.max_annotation_strings = 1024 * 1024;
  options.num_gpus = num_gpus;
  return options;
}

absl::Status GpuTracer::DoStart() {
  if (!rocm_tracer_->IsAvailable()) {
    return tsl::errors::Unavailable("Another profile session running.");
  }
  
  // AnnotationStack::Enable(true);

  RocmTraceCollectorOptions trace_collector_options =
      GetRocmTraceCollectorOptions(rocm_tracer_->NumGpus());
  uint64_t start_gputime_ns = rocm_tracer_->GetTimestamp();
  uint64_t start_walltime_ns = tsl::EnvTime::NowNanos();
  rocm_trace_collector_ = CreateRocmCollector(
      trace_collector_options, start_walltime_ns, start_gputime_ns);

  // RocmTracerOptions tracer_options = GetRocmTracerOptions();
  // rocm_tracer_->Enable(tracer_options, rocm_trace_collector_.get());
  
  // LOG(ERROR) << "cj rocm_tracer_collector = " << rocm_trace_collector_.get();
  // LOG(ERROR) << "cj rocm_tracer_ collector = " << rocm_tracer_->get_collector();
  // LOG(ERROR) << "cj check XSpace = " << space;
  LOG(ERROR) << "DO START ...";

  /*
  RocmTracer* rocm_tracer_ =
      profiler::RocmTracer::GetRocmTracerSingleton();
  LOG(ERROR) << "cj rocm_tracer is available = " << rocm_tracer_->IsAvailable();
  if (!rocm_tracer_->IsAvailable()) {
    return absel::;
  }
  */
 for (auto& event: rocm_tracer_->GetEvents()) {
  rocm_trace_collector_->AddEvent(std::move(event));
 }
 LOG(ERROR) << "DO START after moving events...";
 rocm_trace_collector_->Flush();
 LOG(ERROR) << "DO START after flush...";

 LOG(ERROR) << "Export XSpace after flush...";
  return absl::OkStatus();
}

absl::Status GpuTracer::Start() {
  absl::Status status = DoStart();
  if (status.ok()) {
    profiling_state_ = State::kStartedOk;
    return absl::OkStatus();
  } else {
    profiling_state_ = State::kStartedError;
    return status;
  }
}

absl::Status GpuTracer::DoStop() {
  // rocm_tracer_->stop();
  // rocm_tracer_->shutdown();
  return absl::OkStatus();  
}

absl::Status GpuTracer::Stop() {
  if (profiling_state_ == State::kStartedOk) {
    absl::Status status = DoStop();
    profiling_state_ = status.ok() ? State::kStoppedOk : State::kStoppedError;
  }
  return absl::OkStatus();
}

absl::Status GpuTracer::CollectData(XSpace* space) {
  if (rocm_trace_collector_) rocm_trace_collector_->Export(space);
  LOG(ERROR) << "CollectData XSpace = " << space;

  switch (profiling_state_) {
    case State::kNotStarted:
      VLOG(3) << "No trace data collected, session wasn't started";
      return absl::OkStatus();
    case State::kStartedOk:
      return tsl::errors::FailedPrecondition(
          "Cannot collect trace before stopping");
    case State::kStartedError:
      LOG(ERROR) << "Cannot collect, roctracer failed to start";
      return absl::OkStatus();
    case State::kStoppedError:
      VLOG(3) << "No trace data collected";
      return absl::OkStatus();
    case State::kStoppedOk: {
      // if (rocm_trace_collector_) rocm_trace_collector_->Export(space);
      return absl::OkStatus();
    }
  }
  return tsl::errors::Internal("Invalid profiling state: ", profiling_state_);
}

// Not in anonymous namespace for testing purposes.
std::unique_ptr<profiler::ProfilerInterface> CreateGpuTracer(
    const ProfileOptions& options) {
  if (options.device_type() != ProfileOptions::GPU &&
      options.device_type() != ProfileOptions::UNSPECIFIED){
    return nullptr;
  }

  profiler::RocmTracer* rocm_tracer =
      profiler::RocmTracer::GetRocmTracerSingleton();
  LOG(ERROR) << "cj rocm_tracer is available = " << rocm_tracer->IsAvailable();
  LOG(ERROR) << "Traced events = " << rocm_tracer->GetEvents().size();
  if (!rocm_tracer->IsAvailable()) {
    return nullptr;
  }

  return std::make_unique<profiler::GpuTracer>(rocm_tracer);
}

auto register_rocm_gpu_tracer_factory = [] {
  RegisterProfilerFactory(&CreateGpuTracer);
  return 0;
}();

}  // namespace profiler
}  // namespace xla

#endif  // TENSORFLOW_USE_ROCM

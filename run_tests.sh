#!/bin/bash
set +e # ignore errors
rm /var/lock/gpu*
bazel test --config=rocm --nocache_test_results --flaky_test_attempts=1 --jobs=32 --run_under=//tools/ci_build/gpu_build:parallel_gpu_execute -k \
    --test_tag_filters=-no_oss,-no_rocm,-requires-gpu-nvidia,-requires-gpu-sm70 \
    --build_tag_filters=-no_oss,-no_rocm,-requires-gpu-nvidia,-requires-gpu-sm70 -- //xla/... -//xla/stream_executor/cuda/...

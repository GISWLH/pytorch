#pragma once

#include <c10/core/Allocator.h>
#include <c10/cuda/CUDAStream.h>

namespace c10::cuda::CUDAGraphMemory {

struct CaptureInfo {
  cudaGraph_t graph{};
  CaptureId_t capture_id{0};
  const cudaGraphNode_t* terminals{nullptr};
  size_t num_terminals{0};
  cudaStreamCaptureStatus status{cudaStreamCaptureStatusNone};
};

C10_CUDA_API CaptureInfo
getCaptureInfo(cudaStream_t stream, bool include_dependencies = true);
C10_CUDA_API void waitStream(CUDAStream waiter, CUDAStream source);

// Owns CUDA Graph-specific capture state for the native allocator. Block
// lifecycle and conditional-capture policy are added only when CCA starts
// delegating them.
class CaptureTracker {
 public:
  void captureBegin();
  void captureEnd();
  bool hasActiveCaptures() const {
    return active_captures_ != 0;
  }

 private:
  int active_captures_{0};
};

} // namespace c10::cuda::CUDAGraphMemory

#include <c10/cuda/impl/CUDAGraphMemory.h>

#include <c10/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <c10/util/SmallVector.h>

namespace c10::cuda::CUDAGraphMemory {

CaptureInfo getCaptureInfo(cudaStream_t stream, bool include_dependencies) {
  CaptureInfo info;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 13000
  C10_CUDA_CHECK(cudaStreamGetCaptureInfo(
      stream,
      &info.status,
      &info.capture_id,
      &info.graph,
      include_dependencies ? &info.terminals : nullptr,
      nullptr,
      include_dependencies ? &info.num_terminals : nullptr));
#else
  C10_CUDA_CHECK(cudaStreamGetCaptureInfo_v2(
      stream,
      &info.status,
      &info.capture_id,
      &info.graph,
      include_dependencies ? &info.terminals : nullptr,
      include_dependencies ? &info.num_terminals : nullptr));
#endif
  TORCH_CHECK(
      info.status != cudaStreamCaptureStatusInvalidated,
      "Cannot establish a dependency in an invalidated CUDA graph capture");
  if (info.status == cudaStreamCaptureStatusNone) {
    return {};
  }
  return info;
}

void waitStream(CUDAStream waiter, CUDAStream source) {
  TORCH_CHECK(waiter.device() == source.device());
  if (waiter.stream() == source.stream()) {
    return;
  }
  CUDAGuard guard(source.device());
  auto source_info = getCaptureInfo(source);
  // CUDA owns the terminal array; copy it before inspecting another stream.
  SmallVector<cudaGraphNode_t> dependencies;
  for (size_t i = 0; i < source_info.num_terminals; ++i) {
    dependencies.push_back(source_info.terminals[i]);
  }
  auto waiter_info = getCaptureInfo(waiter);
  if (source_info.status == cudaStreamCaptureStatusNone &&
      waiter_info.status == cudaStreamCaptureStatusNone) {
    CUDAEvent event;
    event.record(source);
    event.block(waiter);
    return;
  }
  TORCH_CHECK(
      source_info.status == cudaStreamCaptureStatusActive &&
          waiter_info.status == cudaStreamCaptureStatusActive &&
          source_info.capture_id == waiter_info.capture_id &&
          source_info.graph == waiter_info.graph,
      "Cannot establish a COW stream dependency across CUDA graph capture "
      "boundaries; order the operation outside capture first");
  if (dependencies.empty()) {
    return;
  }
#if defined(CUDA_VERSION) && CUDA_VERSION >= 13000
  C10_CUDA_CHECK(cudaStreamUpdateCaptureDependencies(
      waiter,
      dependencies.data(),
      nullptr,
      dependencies.size(),
      cudaStreamAddCaptureDependencies));
#else
  C10_CUDA_CHECK(cudaStreamUpdateCaptureDependencies(
      waiter,
      dependencies.data(),
      dependencies.size(),
      cudaStreamAddCaptureDependencies));
#endif
}

void CaptureTracker::captureBegin() {
  ++active_captures_;
}

void CaptureTracker::captureEnd() {
  TORCH_INTERNAL_ASSERT(
      active_captures_ > 0, "captureEnd called with no active capture");
  --active_captures_;
}

} // namespace c10::cuda::CUDAGraphMemory

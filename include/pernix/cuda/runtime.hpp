#ifndef PERNIX_CUDA_RUNTIME
#define PERNIX_CUDA_RUNTIME

#if defined(_WIN32) && defined(PERNIX_CUDA_SHARED)
#if defined(PERNIX_CUDA_BUILD)
#define PERNIX_CUDA_API __declspec(dllexport)
#else
#define PERNIX_CUDA_API __declspec(dllimport)
#endif
#else
#define PERNIX_CUDA_API
#endif

namespace pernix::cuda {

PERNIX_CUDA_API int device_count() noexcept;

}  // namespace pernix::cuda

#endif  // PERNIX_CUDA_RUNTIME

#include <pernix/cuda/runtime.hpp>

#include <cuda_runtime_api.h>

namespace pernix::cuda {

int device_count() noexcept {
    int count = 0;

    const cudaError_t result = cudaGetDeviceCount(&count);

    if (result != cudaSuccess) {
        return 0;
    }

    return count;
}

} // namespace pernix::cuda

#ifndef PERNIX_CUDA_CODEC
#define PERNIX_CUDA_CODEC

#include <pernix/compat.h>
#include <pernix/pernix.h>
#include <pernix/cuda/runtime.hpp>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace pernix::cuda {

PERNIX_CUDA_API cudaError_t decompress_blocks_f32(
    const u8* compressed_data,
    f32* decompressed_data,
    usize blocks,
    u8 bit_width,
    f32 scale,
    bool sign_values = true,
    cudaStream_t stream = nullptr
) noexcept;

// Host-buffer adapter used by the C++/C dispatchers. It copies the packed input
// to the active CUDA device, launches the decompression kernel, and copies the
// decoded values back to the caller's host buffer.
PERNIX_CUDA_API pernix_status decompress_blocks_f32_host(
    const u8* compressed_data,
    f32* decompressed_data,
    usize blocks,
    u8 bit_width,
    f32 scale,
    bool sign_values = true
) noexcept;

}  // namespace pernix::cuda

#endif  // PERNIX_CUDA_CODEC

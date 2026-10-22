#include <pernix/cuda/codec.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace pernix::cuda {
namespace {

constexpr unsigned block_bytes = 64;
constexpr unsigned block_bits = block_bytes * 8;
constexpr unsigned warp_size = 32;
constexpr unsigned warps_per_cta = 4;

struct DeviceBuffer {
private:
    void* pointer = nullptr;

public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (pointer != nullptr) {
            (void)cudaFree(pointer);
        }
    }

    cudaError_t allocate(const usize bytes) noexcept {
        return cudaMalloc(&pointer, bytes);
    }
};

pernix_status status_from_cuda_error(const cudaError_t error) noexcept {
    if (error == cudaSuccess) {
        return PERNIX_STATUS_OK;
    }
    if (error == cudaErrorMemoryAllocation) {
        return PERNIX_STATUS_OUT_OF_MEMORY;
    }
    if (error == cudaErrorInvalidValue) {
        return PERNIX_STATUS_INVALID_ARGUMENT;
    }
    return PERNIX_STATUS_UNSUPPORTED_IMPLEMENTATION;
}

__device__ __forceinline__ u32 unpack_value(const u8* compressed_data, const usize idx, const u8 bit_width) {
    const usize bit_pos    = idx * bit_width;
    const usize byte_pos   = bit_pos >> 3;
    const usize bit_offset = bit_pos & 7u;

    u32 word = 0;

#pragma unroll
    for (usize i = 0; i < 4; ++i) {
        if (byte_pos + i < block_bytes) {
            word |= static_cast<u32>(compressed_data[byte_pos + i]) << (i * 8);
        }
    }

    const u32 mask = (1u << bit_width) - 1u;

    return (word >> bit_offset) & mask;
}

__device__ __forceinline__
i32 decode_signed(const u32 raw_value, const u8 bit_width) {
    const std::uint32_t sign_bit = 1u << (bit_width - 1);

    return static_cast<std::int32_t>((raw_value ^ sign_bit) - sign_bit);
}
}

__global__
void decompress_kernel_f32(const u8* __restrict__ compressed_data, f32* __restrict__ decompressed_data, usize blocks, u8 bit_width, f32 scale, bool sign_values) {
    const usize warp_in_cta = threadIdx.x / warp_size;
    const usize lane = threadIdx.x % warp_size;
    const usize warp_index = blockIdx.x * warps_per_cta + warp_in_cta;
    const usize warp_stride = gridDim.x * warps_per_cta;
    const usize elements_per_block = block_bits / bit_width;

    for (usize block = warp_index; block < blocks; block += warp_stride) {
        const u8* block_input = compressed_data + block * block_bytes;
        f32* block_output = decompressed_data + block * elements_per_block;
        for (usize i = lane; i < elements_per_block; i += warp_size) {
            const u32 raw_value = unpack_value(block_input, i, bit_width);
            const i32 signed_value = sign_values ? decode_signed(raw_value, bit_width) : static_cast<i32>(raw_value);
            block_output[i] = static_cast<f32>(signed_value) * scale;
        }
    }

}


cudaError_t decompress_blocks_f32(const u8* compressed_data, f32* decompressed_data, usize blocks, u8 bit_width, f32 scale,
                                         bool sign_values, cudaStream_t stream) noexcept {
    if (compressed_data == nullptr ||
        decompressed_data == nullptr ||
        blocks == 0 ||
        bit_width < 1 ||
        bit_width > 24 ||
        !std::isfinite(scale) ||
        scale <= 0.0f)
    {
        return cudaErrorInvalidValue;
    }

    const usize elements_per_block = block_bits / bit_width;

    if (blocks > std::numeric_limits<usize>::max() / block_bytes ||
        blocks > std::numeric_limits<usize>::max() / elements_per_block ||
        blocks * elements_per_block > std::numeric_limits<usize>::max() / sizeof(f32)) {
        return cudaErrorInvalidValue;
    }

    int device = 0;
    if (const cudaError_t error = cudaGetDevice(&device); error != cudaSuccess) {
        return error;
    }

    cudaDeviceProp properties{};
    if (const cudaError_t error = cudaGetDeviceProperties(&properties, device); error != cudaSuccess) {
        return error;
    }

    const std::size_t requested_ctas = blocks / warps_per_cta + (blocks % warps_per_cta != 0);

    const std::size_t max_ctas = static_cast<std::size_t>(properties.multiProcessorCount) * 8;

    const unsigned grid_size = static_cast<unsigned>(requested_ctas < max_ctas ? requested_ctas : max_ctas);

    constexpr unsigned threads_per_cta = warps_per_cta * warp_size;

    decompress_kernel_f32<<<grid_size, threads_per_cta, 0, stream>>>(compressed_data, decompressed_data, blocks, bit_width, scale, sign_values);

    return cudaGetLastError();
}

pernix_status decompress_blocks_f32_host(const u8* compressed_data, f32* decompressed_data, const usize blocks, const u8 bit_width,
                                          const f32 scale, const bool sign_values) noexcept {
    if (compressed_data == nullptr || decompressed_data == nullptr || blocks == 0 || bit_width < 1 || bit_width > 24 ||
        !std::isfinite(scale) || scale <= 0.0f) {
        return PERNIX_STATUS_INVALID_ARGUMENT;
    }

    const usize elements_per_block = block_bits / bit_width;
    const usize max_size = std::numeric_limits<usize>::max();
    if (blocks > max_size / block_bytes || blocks > max_size / elements_per_block ||
        blocks * elements_per_block > max_size / sizeof(f32)) {
        return PERNIX_STATUS_INVALID_ARGUMENT;
    }

    const usize compressed_bytes = blocks * block_bytes;
    const usize output_bytes = blocks * elements_per_block * sizeof(f32);
    DeviceBuffer device_input;
    DeviceBuffer device_output;

    if (const auto status = status_from_cuda_error(device_input.allocate(compressed_bytes)); status != PERNIX_STATUS_OK) {
        return status;
    }
    if (const auto status = status_from_cuda_error(device_output.allocate(output_bytes)); status != PERNIX_STATUS_OK) {
        return status;
    }
    if (const auto status = status_from_cuda_error(
            cudaMemcpy(device_input.pointer, compressed_data, compressed_bytes, cudaMemcpyHostToDevice));
        status != PERNIX_STATUS_OK) {
        return status;
    }

    const cudaError_t launch_status = decompress_blocks_f32(static_cast<const u8*>(device_input.pointer),
                                                             static_cast<f32*>(device_output.pointer), blocks, bit_width, scale,
                                                             sign_values, nullptr);
    if (const auto status = status_from_cuda_error(launch_status); status != PERNIX_STATUS_OK) {
        return status;
    }

    return status_from_cuda_error(cudaMemcpy(decompressed_data, device_output.pointer, output_bytes, cudaMemcpyDeviceToHost));
}
}

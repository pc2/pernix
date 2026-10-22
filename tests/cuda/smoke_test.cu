#include <cuda_runtime.h>
#include <pernix/cuda/runtime.hpp>

#include <iostream>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        const cudaError_t error = (call);                     \
        if (error != cudaSuccess) {                           \
            throw std::runtime_error(                         \
                std::string(#call) + ": " +                   \
                cudaGetErrorString(error));                   \
        }                                                     \
    } while (false)

__global__ static void smoke_kernel(int* output) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *output = 42;
    }
}
namespace {

class DeviceInt {
public:
    DeviceInt() {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), sizeof(*data_)));
    }

    DeviceInt(const DeviceInt&) = delete;
    DeviceInt& operator=(const DeviceInt&) = delete;

    ~DeviceInt() {
        if (data_ != nullptr) {
            (void)cudaFree(data_);
        }
    }

    [[nodiscard]] int* get() const noexcept {
        return data_;
    }

    cudaError_t release() noexcept {
        if (data_ == nullptr) {
            return cudaSuccess;
        }

        int* const data = data_;
        data_ = nullptr;
        return cudaFree(data);
    }

private:
    int* data_ = nullptr;
};

}  // namespace

int main() {
    try {
        int device_count = 0;
        const cudaError_t count_result = cudaGetDeviceCount(&device_count);
        if (count_result != cudaSuccess) {
            std::cerr << "CUDA smoke test skipped: " << cudaGetErrorString(count_result) << '\n';
            return 77;
        }
        if (device_count == 0) {
            std::cerr << "CUDA smoke test skipped: no CUDA-capable devices were detected\n";
            return 77;
        }

        CUDA_CHECK(cudaSetDevice(0));

        const int backend_device_count = pernix::cuda::device_count();
        if (backend_device_count != device_count) {
            throw std::runtime_error("CUDA backend device count does not match the runtime");
        }

        cudaDeviceProp device_properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&device_properties, 0));

        DeviceInt device_output;
        smoke_kernel<<<1, 1>>>(device_output.get());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        int host_output = 0;
        CUDA_CHECK(cudaMemcpy(&host_output, device_output.get(), sizeof(host_output), cudaMemcpyDeviceToHost));
        if (host_output != 42) {
            throw std::runtime_error("smoke_kernel returned " + std::to_string(host_output) + ", expected 42");
        }

        CUDA_CHECK(device_output.release());
        std::cout << "CUDA smoke test passed on " << device_properties.name << " (" << device_count
                  << " device(s)); kernel output: " << host_output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CUDA smoke test failed: " << error.what() << '\n';
        return 1;
    }
}

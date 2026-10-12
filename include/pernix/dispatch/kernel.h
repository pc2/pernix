#ifndef PERNIX_KERNEL_H
#define PERNIX_KERNEL_H

#include <pernix/compat.h>

namespace pernix::internal {
using KernelBlockF32Func  = i32 (*)(const void*, f32, void*);
using KernelBlocksF32Func = i32 (*)(const void*, f32, void*, u32);
using KernelBlockF64Func  = i32 (*)(const void*, f64, void*);
using KernelBlocksF64Func = i32 (*)(const void*, f64, void*, u32);

template <typename FuncType>
struct Kernel {
    FuncType func;
    pernix_status status;

    explicit operator bool() const noexcept { return func != nullptr; }

    constexpr Kernel(FuncType kernel_func)
        : func(kernel_func), status(kernel_func != nullptr ? PERNIX_STATUS_OK : PERNIX_STATUS_UNSUPPORTED_BIT_WIDTH) {}

    constexpr Kernel(const pernix_status kernel_status, FuncType kernel_func = nullptr)
        : func(kernel_func), status(kernel_func != nullptr ? PERNIX_STATUS_OK : kernel_status) {}

    static constexpr Kernel failure(const pernix_status kernel_status) { return Kernel(kernel_status); }
};
}  // namespace pernix::internal

#endif  // PERNIX_KERNEL_H

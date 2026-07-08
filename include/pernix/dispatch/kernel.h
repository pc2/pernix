#ifndef PERNIX_KERNEL_H
#define PERNIX_KERNEL_H

#include <pernix/compat.h>
#include <string_view>

namespace pernix::internal {
using KernelBlockF32Func  = i32 (*)(const void*, f32, void*);
using KernelBlocksF32Func = i32 (*)(const void*, f32, void*, u32);
using KernelBlockF64Func  = i32 (*)(const void*, f64, void*);
using KernelBlocksF64Func = i32 (*)(const void*, f64, void*, u32);

template <typename FuncType>
struct Kernel {
    std::string_view name;
    FuncType func;

    explicit operator bool() const noexcept {
        return func != nullptr;
    }

    Kernel(const std::string_view kernel_name, FuncType kernel_func) : name(kernel_name), func(kernel_func) {
    }
};
}

#endif //PERNIX_KERNEL_H

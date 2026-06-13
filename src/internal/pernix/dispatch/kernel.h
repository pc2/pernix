#ifndef PERNIX_KERNEL_H
#define PERNIX_KERNEL_H

#include <cstdint>
#include <string_view>

namespace pernix::internal {
using KernelBlockF32Func  = int (*)(const void*, float, void*);
using KernelBlocksF32Func = int (*)(const void*, float, void*, unsigned int);
using KernelBlockF64Func  = int (*)(const void*, double, void*);
using KernelBlocksF64Func = int (*)(const void*, double, void*, unsigned int);

template <typename FuncType>
struct Kernel {
    std::string_view name;
    FuncType func;

    explicit operator bool() const noexcept {
        return func != nullptr;
    }

    Kernel(const std::string_view name, FuncType func) : name(name), func(func) {
    }
};
}

#endif //PERNIX_KERNEL_H

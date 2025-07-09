/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: Allocator
 * Author: KPL
 * Create: 2024-04-10
 * Notes: NA
 */

#ifndef KDNN_SERVICE_HPP
#define KDNN_SERVICE_HPP

#include <type_traits>
#include <limits>
#include <cstddef>

#include "types/kdnn_types.hpp"
#include "types/kdnn_data_type.hpp"
#include "types/kdnn_layout.hpp"
#include "service/kdnn_exception.hpp"

namespace KDNN {
namespace Service {

constexpr SizeType ALIGNMENT = 128;

KDNN_API_PUBLIC SizeType GetIdxForUserLayout(Layout l, SizeType idx) noexcept(false);

KDNN_API_PUBLIC Shape GetShapeAccordingToLayout(const Shape &sh, Layout parLayout) noexcept;

KDNN_API_PUBLIC Shape FlushStrides(const Shape &dims, const Shape &strides) noexcept(false);

template<typename...>
struct IsAllSame : std::false_type {};

template<>
struct IsAllSame<> : std::true_type {};

template<typename T>
struct IsAllSame<T> : std::true_type {};

template<typename T>
struct IsAllSame<T, T> : std::true_type {};

template<typename T, typename... U>
struct IsAllSame<T, T, U...> : IsAllSame<T, U...> {};

template <typename IntType,
    typename std::enable_if<std::is_integral<typename std::remove_reference<IntType>::type>::value,
    bool>::type = true>
bool WillIntMultOverflow(IntType a, IntType b) noexcept
{
    if ((a == 0) || (b == 0)) {
        return false;
    } else {
        volatile IntType mul = a * b;
        return (mul / a) != b;
    }
}

template <typename IntType, typename ... Args,
    typename std::enable_if<IsAllSame<IntType, Args...>::value, bool>::type = true>
bool WillIntMultOverflow(IntType a, IntType b, Args ... args) noexcept
{
    if (WillIntMultOverflow(a, b)) {
        return true;
    } else {
        return WillIntMultOverflow(a * b, args ...);
    }
}

template <typename ForwardIt,
    typename std::enable_if<std::is_integral<typename std::iterator_traits<ForwardIt>::value_type>::value,
    bool>::type = true>
bool WillIntMultOverflow(ForwardIt begin, ForwardIt end) noexcept
{
    typename std::iterator_traits<ForwardIt>::value_type mult = 1;
    for (auto &&it = begin; it != end; ++it) {
        if (WillIntMultOverflow(mult, *it)) {
            return true;
        }
        mult *= *it;
    }
    return false;
}

// User may use allocation/deallocation functions directly or via allocator class
KDNN_API_PUBLIC void *AlignedAlloc(SizeType n, SizeType alignment = ALIGNMENT) noexcept(false);
KDNN_API_PUBLIC void Deallocate(void *p, SizeType n = 0) noexcept;

// This class represents an allocator which user may use in order to obtain aligned memory
// and pass it inside KDNN functions. To be compatible with std::allocator all member names are in snake_case
template <typename T, SizeType alignment = ALIGNMENT>
struct AlignedAllocator {
    using value_type = T;
    using pointer = T *;
    using const_pointer = const T *;
    using reference = T &;
    using const_reference = const T &;
    using size_type = ::KDNN::SizeType;
    using difference_type = std::ptrdiff_t;

    template <typename U> struct rebind {
        using other = AlignedAllocator<U, alignment>;
    };

    T *allocate(SizeType n) const noexcept(false)
    {
        if (n > std::numeric_limits<SizeType>::max() / sizeof(T)) {
            throw BadArrayNewLength();
        }
        return static_cast<T*>(AlignedAlloc(n * sizeof(T), alignment));
    }

    void deallocate(T *p, SizeType n = 0) const noexcept
    {
        ::KDNN::Service::Deallocate(p, n);
    }
    virtual ~AlignedAllocator() = default;
};

template <typename T>
struct Deallocator {
    void operator()(T *p, ::KDNN::SizeType n = 0) const noexcept
    {
        ::KDNN::Service::Deallocate(p, n);
    }
};

template <typename T, SizeType alignment_1, typename U, SizeType alignment_2>
constexpr bool operator == (const AlignedAllocator<T, alignment_1> &,
    const AlignedAllocator<U, alignment_2> &) noexcept
{
    return (alignment_1 == alignment_2) && std::is_same<T, U>::value;
}
template <typename T, SizeType alignment_1, typename U, SizeType alignment_2>
constexpr bool operator != (const AlignedAllocator<T, alignment_1> &lhs,
    const AlignedAllocator<U, alignment_2> &rhs) noexcept
{
    return !(lhs == rhs);
}

} // Service
} // KDNN

#endif // KDNN_SERVICE_HPP

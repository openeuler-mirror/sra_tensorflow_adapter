/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: Provide runtime info about data type.
 * Author: KPL
 * Create: 2024-04-10
 * Notes: NA
 */

#ifndef KDNN_DATATYPE_HPP
#define KDNN_DATATYPE_HPP

#include <cstddef>
#include <cstdint>

#include "service/kdnn_exception.hpp"

#define KDNN_API_PUBLIC __attribute__((visibility("default")))

#define GCC_VER (__GNUC__ * 100 + __GNUC_MINOR__)

#if GCC_VER >= 1000
typedef __fp16 fp16_t;
#else
struct fp16_t {
    uint16_t data;
};
#endif

#if GCC_VER >= 1000
typedef __bf16 bf16_t;
#else
struct bf16_t {
    uint16_t data;
};
#endif

namespace KDNN {

using SizeType = std::size_t;
using IntType = std::int64_t;

namespace Element {

// enumeration of KDNN supported data types
enum class TypeT {
    UNDEFINED = 0,
    F32,
    F16,
    BF16,
    S32,
    S8,
    U8
};

// This class is designed to provide runtime info about data type.
struct Type final {
public:
    using SizeType = ::KDNN::SizeType;
    Type(const TypeT& t) noexcept(false) : type{t}, size(0)
    {
        switch (type) {
            case TypeT::F32: {
                size = sizeof(float);
                break;
            }
            case TypeT::F16: {
                size = sizeof(fp16_t);
                break;
            }
            case TypeT::BF16: {
                size = sizeof(bf16_t);
                break;
            }
            case TypeT::S32: {
                size = sizeof(std::int32_t);
                break;
            }
            case TypeT::S8: {
                size = sizeof(std::int8_t);
                break;
            }
            case TypeT::U8: {
                size = sizeof(std::uint8_t);
                break;
            }
            default: {}
        }
        if (type == TypeT::UNDEFINED) {
            // throw Service::LogicError {"Type: unsupported data type"};
            size = 0;
        }
    }
    SizeType GetSize() const noexcept
    {
        return size;
    }
    SizeType GetBitwidth() const noexcept
    {
        return (GetSize() * 8uLL);
    }
    operator TypeT() const noexcept
    {
        return type;
    }
    bool IsFP() const noexcept
    {
        if ((type == TypeT::F32) ||
            (type == TypeT::F16) ||
            (type == TypeT::BF16)) {
            return true;
        } else {
            return false;
        }
    }
    bool IsIntegral() const noexcept
    {
        return !IsFP();
    }
    bool IsSigned() const noexcept
    {
        if (type == TypeT::U8) {
            return false;
        } else {
            return true;
        }
    }
    bool IsUnsigned() const noexcept
    {
        return !IsSigned();
    }
private:
    TypeT type;
    SizeType size;
};

template <typename ElementType>
inline Type MatchType()
{
    return TypeT::UNDEFINED;
}

template <>
inline Type MatchType<float>()
{
    return TypeT::F32;
}

template <>
inline Type MatchType<fp16_t>()
{
    return TypeT::F16;
}

template <>
inline Type MatchType<bf16_t>()
{
    return TypeT::BF16;
}

template <>
inline Type MatchType<std::int32_t>()
{
    return TypeT::S32;
}

template <>
inline Type MatchType<std::int8_t>()
{
    return TypeT::S8;
}

template <>
inline Type MatchType<std::uint8_t>()
{
    return TypeT::U8;
}

inline bool operator==(const Type &lhs, const TypeT &rhs) noexcept
{
    return (static_cast<TypeT>(lhs) == rhs);
}

inline bool operator==(const TypeT &lhs, const Type &rhs) noexcept
{
    return (lhs == static_cast<TypeT>(rhs));
}

inline bool operator==(const Type &lhs, const Type &rhs) noexcept
{
    return (lhs.GetSize() == rhs.GetSize()) &&
           (static_cast<TypeT>(lhs) == (static_cast<TypeT>(rhs)));
}

inline bool operator!=(const Type &lhs, const TypeT &rhs) noexcept
{
    return !(lhs == rhs);
}
inline bool operator!=(const TypeT &lhs, const Type &rhs) noexcept
{
    return !(lhs == rhs);
}
inline bool operator!=(const Type &lhs, const Type &rhs) noexcept
{
    return !(lhs == rhs);
}

} // Element
} // KDNN

#endif // KDNN_DATATYPE_HPP

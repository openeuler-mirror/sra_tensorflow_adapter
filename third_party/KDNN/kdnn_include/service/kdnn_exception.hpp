/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: Exceptions
 * Author: KPL
 * Create: 2024-04-10
 * Notes: NA
 */

#ifndef KDNN_EXCEPTION_HPP
#define KDNN_EXCEPTION_HPP

#include <exception>
#include <new>
#include <stdexcept>

namespace KDNN {
namespace Service {

// Error handling in KDNN is performed via exceptions.
// KDNN has its own exceptions which are nested from std::exception standard class
struct BadAlloc : public std::bad_alloc {
    const char* what() const noexcept override
    {
        return "KDNN allocation failed";
    }
};

struct BadArrayNewLength : public std::bad_array_new_length {
    const char* what() const noexcept override
    {
        return "KDNN bad array new length";
    }
};

struct LogicError : public std::logic_error {
    explicit LogicError(const std::string& whatArg) : std::logic_error(whatArg) {}
    explicit LogicError(const char* whatArg) : std::logic_error(whatArg) {}
    LogicError(const LogicError& other) noexcept : std::logic_error(other) {}
    LogicError& operator=(const LogicError& other) noexcept = default;
};

struct Unsupported : public std::invalid_argument {
    explicit Unsupported(const std::string& whatArg) : std::invalid_argument(whatArg) {}
    explicit Unsupported(const char* whatArg) : std::invalid_argument(whatArg) {}
    Unsupported(const Unsupported& other) noexcept : std::invalid_argument(other) {}
    Unsupported& operator=(const Unsupported& other) noexcept = default;
};

} // Service
} // KDNN

#endif // KDNN_EXCEPTION_HPP

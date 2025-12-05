// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>
#include <utility>

namespace hclpp {

template <class R, class T>
R int_cast(T t)
{
    if (std::in_range<R>(t)) {
        return static_cast<R>(t);
    }
    throw std::runtime_error("Conversion cannot preserve value representation");
}

} // namespace hclpp

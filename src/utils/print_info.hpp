// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>
#include <ios>
#include <ostream>
#include <string_view>

namespace novapp {

template <class T>
void print_info(std::ostream& os, std::string_view const var_name, T const& var_value)
{
    os << std::left << std::setw(40) << std::setfill('.') << var_name;
    os << std::right << std::setw(40) << std::setfill('.') << var_value << '\n';
}

} // namespace novapp

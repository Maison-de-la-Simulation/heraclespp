// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file integration.hpp
//!

#pragma once

#include <kokkos_shortcut.hpp>

namespace hclpp {

class Grid;
class Range;

[[nodiscard]] auto integrate(Range const& range, Grid const& grid, KV_cdouble_3d const& var) -> double;

} // namespace hclpp

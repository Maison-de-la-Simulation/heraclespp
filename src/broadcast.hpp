// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <kokkos_shortcut.hpp>

namespace hclpp {

class Range;

void broadcast(Range const& range, double in, KV_double_3d const& out);

void broadcast(Range const& range, double in, KV_double_4d const& out);

void broadcast(Range const& range, KV_cdouble_1d const& in, KV_double_3d const& out);

void broadcast(Range const& range, KV_cdouble_1d const& in, KV_double_4d const& out);

} // namespace hclpp

// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file geometry_factory.hpp
//!

#pragma once

#include <memory>

namespace hclpp {

class IComputeGeom;

auto factory_grid_geometry() -> std::unique_ptr<IComputeGeom>;

} // namespace hclpp

// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file geometry.hpp
//!

#pragma once

#include <memory>

namespace novapp
{

class IComputeGeom;

std::unique_ptr<IComputeGeom> factory_grid_geometry();

} // namespace novapp

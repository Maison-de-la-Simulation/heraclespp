// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file grid_factory.hpp
//!

#pragma once

#include <memory>
#include <string>

namespace novapp
{

class IGridType;
class Param;

std::unique_ptr<IGridType> factory_grid_type(
    std::string const& grid,
    Param const& param);

} // namespace novapp

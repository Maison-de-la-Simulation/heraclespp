//!
//! @file grid_factory.hpp
//!

#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include "mesh/grid_type.hpp"
#include "nova_params.hpp"

namespace novapp
{

inline std::unique_ptr<IGridType> factory_grid_type(
    std::string const& grid,
    Param const& param)
{
    if (grid == "Regular")
    {
        return std::make_unique<Regular>(param);
    }
    if (grid == "UserDefined")
    {
        return std::make_unique<GridSetup>(param);
    }
    throw std::runtime_error("Unknown grid type : " + grid + ".");
}

} // namespace novapp

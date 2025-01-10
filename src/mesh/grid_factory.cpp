//!
//! @file grid_factory.cpp
//!

#include <memory>
#include <stdexcept>
#include <string>

#include "grid_factory.hpp"
#include "grid_type.hpp"

namespace novapp
{

class Param;

std::unique_ptr<IGridType> factory_grid_type(
    std::string const& grid,
    Param const& param)
{
    if (grid == "Regular")
    {
        return std::make_unique<Regular>(param);
    }

    throw std::runtime_error("Unknown grid type : " + grid + ".");
}

} // namespace novapp

//!
//! @file grid_factory.hpp
//!

#pragma once

#include "eos.hpp"
#include "nova_params.hpp"
#include "grid_type.hpp"
#include "grid.hpp"
#include "boundary.hpp"
#include "setup.hpp"

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

template <class Gravity>
inline std::unique_ptr<IBoundaryCondition> factory_boundary_construction(
    std::string const& boundary,
    int idim,
    int iface,
    EOS const& eos,
    Grid const& grid,
    ParamSetup const& param_setup,
    Gravity const gravity)
{
    if (boundary == "NullGradient")
    {
        return std::make_unique<NullGradient>(idim, iface, grid);
    }
    if (boundary == "Periodic")
    {
        return std::make_unique<PeriodicCondition>(idim, iface);
    }
    if (boundary == "Reflexive")
    {
        return std::make_unique<ReflexiveCondition>(idim, iface, grid);
    }
    if (boundary == "UserDefined")
    {
        return std::make_unique<BoundarySetup<Gravity>>(idim, iface, eos, grid, param_setup, gravity);
    }
    throw std::runtime_error("Unknown boundary condition : " + boundary + ".");
}

} // namespace novapp
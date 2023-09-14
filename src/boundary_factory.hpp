//!
//! @file boundary_factory.hpp
//!

#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include "boundary.hpp"
#include "eos.hpp"
#include "mesh/grid.hpp"
#include "setup.hpp"

namespace novapp
{

template <class Gravity>
inline std::unique_ptr<IBoundaryCondition> factory_boundary_construction(
    std::string const& boundary,
    int idim,
    int iface,
    EOS const& eos,
    Grid const& grid,
    ParamSetup const& param_setup,
    Gravity const& gravity)
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

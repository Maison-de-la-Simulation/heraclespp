//!
//! @file boundary_factory.hpp
//!

#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include "boundary.hpp"

namespace novapp
{

template<class Gravity>
std::unique_ptr<IBoundaryCondition<Gravity>> factory_boundary_construction(
    std::string const& boundary,
    int idim,
    int iface)
{
    if (boundary == "NullGradient")
    {
        return std::make_unique<NullGradient<Gravity>>(idim, iface);
    }
    if (boundary == "Periodic")
    {
        return std::make_unique<PeriodicCondition<Gravity>>(idim, iface);
    }
    if (boundary == "Reflexive")
    {
        return std::make_unique<ReflexiveCondition<Gravity>>(idim, iface);
    }
    throw std::runtime_error("Unknown boundary condition : " + boundary + ".");
}

} // namespace novapp

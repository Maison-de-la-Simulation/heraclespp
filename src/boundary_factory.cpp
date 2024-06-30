//!
//! @file boundary_factory.cpp
//!

#include <memory>
#include <stdexcept>
#include <string>

#include "boundary.hpp"
#include "boundary_factory.hpp"

namespace novapp
{

std::unique_ptr<IBoundaryCondition> factory_boundary_construction(
    std::string const& boundary,
    int idim,
    int iface)
{
    if (boundary == "NullGradient")
    {
        return std::make_unique<NullGradient>(idim, iface);
    }
    if (boundary == "Periodic")
    {
        return std::make_unique<PeriodicCondition>(idim, iface);
    }
    if (boundary == "Reflexive")
    {
        return std::make_unique<ReflexiveCondition>(idim, iface);
    }
    throw std::runtime_error("Unknown boundary condition : " + boundary + ".");
}

} // namespace novapp

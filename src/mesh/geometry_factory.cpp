//!
//! @file geometry_factory.cpp
//!

#include <memory>
#include <stdexcept>

#include "geom.hpp"
#include "geometry.hpp"

namespace novapp
{

std::unique_ptr<IComputeGeom> factory_grid_geometry()
{
    if (geom_choice == "Cartesian")
    {
        return std::make_unique<Cartesian>();
    }
    if (geom_choice == "Spherical")
    {
        return std::make_unique<Spherical>();
    }
    throw std::runtime_error("Invalid grid geometry: .");
}

} // namespace novapp

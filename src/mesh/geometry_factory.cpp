// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file geometry_factory.cpp
//!

#include <memory>
#include <stdexcept>

#include "geom.hpp"
#include "geometry.hpp"
#include "geometry_factory.hpp"

namespace hclpp {

std::unique_ptr<IComputeGeom> factory_grid_geometry()
{
    if (geom == Geometry::Geom_cartesian) {
        return std::make_unique<Cartesian>();
    }

    if (geom == Geometry::Geom_spherical) {
        return std::make_unique<Spherical>();
    }

    throw std::runtime_error("Invalid grid geometry: .");
}

} // namespace hclpp

//!
//! @file boundary_factory.hpp
//!

#pragma once

#include <memory>
#include <string>

namespace novapp
{

class IBoundaryCondition;

std::unique_ptr<IBoundaryCondition> factory_boundary_construction(
    std::string const& boundary,
    int idim,
    int iface);

} // namespace novapp

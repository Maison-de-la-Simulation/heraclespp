//!
//! @file geometry.hpp
//!

#pragma once

#include <memory>

namespace novapp
{

class IComputeGeom;

std::unique_ptr<IComputeGeom> factory_grid_geometry();

} // namespace novapp

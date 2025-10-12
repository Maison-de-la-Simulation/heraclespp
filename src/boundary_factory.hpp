// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file boundary_factory.hpp
//!

#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include <concepts.hpp>

namespace novapp
{

template <concepts::GravityField Gravity>
class IBoundaryCondition;

template <concepts::GravityField Gravity>
class NullGradient;

template <concepts::GravityField Gravity>
class PeriodicCondition;

template <concepts::GravityField Gravity>
class ReflexiveCondition;

template<concepts::GravityField Gravity>
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

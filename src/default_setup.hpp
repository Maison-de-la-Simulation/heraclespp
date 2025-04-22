// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

# pragma once

#include <stdexcept>

#include <eos.hpp>
#include <kokkos_shortcut.hpp>

#include "default_boundary_setup.hpp"
#include "default_grid_setup.hpp"
#include "default_user_step.hpp"
#include "initialization_interface.hpp"

class INIReader;

namespace novapp
{

class Grid;
class Range;

class ParamSetup
{
public:
    explicit ParamSetup([[maybe_unused]] INIReader const& reader)
    {
    }
};

template <class Gravity>
class InitializationSetup : public IInitializationProblem
{
public:
    InitializationSetup(
        [[maybe_unused]] EOS const& eos,
        [[maybe_unused]] ParamSetup const& param_set_up,
        [[maybe_unused]] Gravity const& gravity)
    {
    }

    void execute(
        [[maybe_unused]] Range const& range,
        [[maybe_unused]] Grid const& grid,
        [[maybe_unused]] KV_double_3d const& rho,
        [[maybe_unused]] KV_double_4d const& u,
        [[maybe_unused]] KV_double_3d const& P,
        [[maybe_unused]] KV_double_4d const& fx) const final
    {
        throw std::runtime_error("Setup not implemented");
    }
};

} // namespace novapp

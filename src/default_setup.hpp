// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>

#include <eos.hpp>
#include <kokkos_shortcut.hpp>

#include "default_boundary_setup.hpp"
#include "default_grid_setup.hpp"
#include "default_user_step.hpp"
#include "initialization_interface.hpp"

class INIReader;

namespace novapp {

class Grid;
class Range;

class ParamSetup
{
public:
    explicit ParamSetup(INIReader const& /*reader*/) {}
};

template <class Gravity>
class InitializationSetup : public IInitializationProblem
{
public:
    InitializationSetup(EOS const& /*eos*/, ParamSetup const& /*param_set_up*/, Gravity const& /*gravity*/) {}

    void execute(
            Range const& /*range*/,
            Grid const& /*grid*/,
            KV_double_3d const& /*rho*/,
            KV_double_4d const& /*u*/,
            KV_double_3d const& /*P*/,
            KV_double_4d const& /*fx*/) const final
    {
        throw std::runtime_error("Setup not implemented");
    }
};

} // namespace novapp

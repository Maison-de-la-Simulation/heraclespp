// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

# pragma once

#include <stdexcept>

#include <eos.hpp>
#include <kokkos_shortcut.hpp>

#include "boundary.hpp"

namespace novapp
{

class Grid;
class ParamSetup;

template <class Gravity>
class BoundarySetup : public IBoundaryCondition<Gravity>
{
public:
    BoundarySetup(int idim, int iface,
        EOS const& /*eos*/,
        ParamSetup const& /*param_setup*/)
        : IBoundaryCondition<Gravity>(idim, iface)
    {
    }

    void execute(Grid const& /*grid*/,
        Gravity const& /*gravity*/,
        KV_double_3d const& /*rho*/,
        KV_double_4d const& /*rhou*/,
        KV_double_3d const& /*E*/,
        KV_double_4d const& /*fx*/) const final
    {
        throw std::runtime_error("Boundary setup not implemented");
    }
};

} // namespace novapp

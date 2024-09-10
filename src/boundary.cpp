//!
//! @file boundary.cpp
//!

#include <array>
#include <cassert>
#include <string>

#include <Kokkos_Core.hpp>
#include <grid.hpp>
#include <kokkos_shortcut.hpp>

#include "boundary.hpp"

namespace novapp
{

std::array<std::string, 3> const bc_dir {"_X", "_Y", "_Z"};
std::array<std::string, 2> const bc_face {"_left", "_right"};


void null_gradient_condition(int m_bc_idim, int m_bc_iface,
                             std::string const& m_label,
                             Grid const& grid,
                             KV_double_3d const& rho,
                             KV_double_4d const& rhou,
                             KV_double_3d const& E,
                             KV_double_4d const& fx)
{
    assert(rho.extent(0) == rhou.extent(0));
    assert(rhou.extent(0) == E.extent(0));
    assert(rho.extent(1) == rhou.extent(1));
    assert(rhou.extent(1) == E.extent(1));
    assert(rho.extent(2) == rhou.extent(2));
    assert(rhou.extent(2) == E.extent(2));

    Kokkos::Array<int, 3> begin {0, 0, 0};
    Kokkos::Array<int, 3> end {rho.extent_int(0), rho.extent_int(1), rho.extent_int(2)};
    int const nfx = fx.extent_int(3);

    int const ng = grid.Nghost[m_bc_idim];
    if (m_bc_iface == 1)
    {
        begin[m_bc_idim] = rho.extent_int(m_bc_idim) - ng;
    }
    end[m_bc_idim] = begin[m_bc_idim] + ng;

    int const offset = m_bc_iface == 0 ? end[m_bc_idim] : begin[m_bc_idim] - 1;
    int const& bc_idim = m_bc_idim;
    Kokkos::parallel_for(
        m_label,
        Kokkos::MDRangePolicy<int, Kokkos::Rank<3>>(begin, end),
        KOKKOS_LAMBDA(int i, int j, int k)
        {
            Kokkos::Array<int, 3> offsets {i, j, k};
            offsets[bc_idim] = offset;
            rho(i, j, k) = rho(offsets[0], offsets[1], offsets[2]);
            for (int n = 0; n < rhou.extent_int(3); ++n)
            {
                rhou(i, j, k, n) = rhou(offsets[0], offsets[1], offsets[2], n);
            }
            E(i, j, k) = E(offsets[0], offsets[1], offsets[2]);
            for (int ifx = 0; ifx < nfx; ++ifx)
            {
                fx(i, j, k, ifx) = fx(offsets[0], offsets[1], offsets[2], ifx);
            }
        });
}


void reflexive_condition(int m_bc_idim, int m_bc_iface,
                         std::string const& m_label,
                         Grid const& grid,
                         KV_double_3d const& rho,
                         KV_double_4d const& rhou,
                         KV_double_3d const& E,
                         KV_double_4d const& fx)
{
    Kokkos::Array<int, 3> begin {0, 0, 0};
    Kokkos::Array<int, 3> end {rho.extent_int(0), rho.extent_int(1), rho.extent_int(2)};
    int const nfx = fx.extent_int(3);

    int const ng = grid.Nghost[m_bc_idim];
    if (m_bc_iface == 1)
    {
        begin[m_bc_idim] = rho.extent_int(m_bc_idim) - ng;
    }
    end[m_bc_idim] = begin[m_bc_idim] + ng;

    int const mirror = m_bc_iface == 0 ? (2 * ng - 1) : (2 * (rho.extent_int(m_bc_idim) - ng) - 1);
    int const& bc_idim = m_bc_idim;
    Kokkos::parallel_for(
        m_label,
        Kokkos::MDRangePolicy<int, Kokkos::Rank<3>>(begin, end),
        KOKKOS_LAMBDA(int i, int j, int k)
        {
            Kokkos::Array<int, 3> offsets {i, j, k};
            offsets[bc_idim] = mirror - offsets[bc_idim];
            rho(i, j, k) = rho(offsets[0], offsets[1], offsets[2]);
            for (int n = 0; n < rhou.extent_int(3); ++n)
            {
                rhou(i, j, k, n) = rhou(offsets[0], offsets[1], offsets[2], n);
            }
            rhou(i, j, k, bc_idim) = -rhou(i, j, k, bc_idim);
            E(i, j, k) = E(offsets[0], offsets[1], offsets[2]);
            for (int ifx = 0; ifx < nfx; ++ifx)
            {
                fx(i, j, k, ifx) = fx(offsets[0], offsets[1], offsets[2], ifx);
            }
        });
}

} // namespace novapp

//!
//! @file boundary.cpp
//!

#include <array>
#include <stdexcept>
#include <string>

#include "boundary.hpp"
#include "grid.hpp"
#include "kokkos_shortcut.hpp"
#include "ndim.hpp"

namespace novapp {

IBoundaryCondition::IBoundaryCondition(int const idim, int const iface) noexcept
    : m_bc_idim(idim)
    , m_bc_iface(iface) {};

IBoundaryCondition::IBoundaryCondition(IBoundaryCondition const& rhs) = default;

IBoundaryCondition::IBoundaryCondition(IBoundaryCondition&& rhs) noexcept = default;

IBoundaryCondition::~IBoundaryCondition() noexcept = default;

IBoundaryCondition& IBoundaryCondition::operator=(IBoundaryCondition const& rhs) = default;

IBoundaryCondition& IBoundaryCondition::operator=(IBoundaryCondition&& rhs) noexcept = default;

void IBoundaryCondition::execute(
        [[maybe_unused]] KV_double_3d const rho,
        [[maybe_unused]] KV_double_4d const rhou,
        [[maybe_unused]] KV_double_3d const E,
        [[maybe_unused]] KV_double_4d const fx) const
{
    throw std::runtime_error("Boundary not implemented");
}

NullGradient::NullGradient(int const idim, int const iface, Grid const& grid)
    : IBoundaryCondition(idim, iface)
    , m_label("NullGradient" + bc_dir[idim] + bc_face[iface])
    , m_grid(std::make_shared<Grid>(grid))
    , m_face{idim, iface}
{
}

void NullGradient::execute(
        KV_double_3d const rho,
        KV_double_4d const rhou,
        KV_double_3d const E,
        KV_double_4d const fx) const noexcept
{
    assert(rho.extent(0) == rhou.extent(0));
    assert(rhou.extent(0) == E.extent(0));
    assert(rho.extent(1) == rhou.extent(1));
    assert(rhou.extent(1) == E.extent(1));
    assert(rho.extent(2) == rhou.extent(2));
    assert(rhou.extent(2) == E.extent(2));

    Kokkos::Array<int, 3> begin {0, 0, 0};
    Kokkos::Array<int, 3> end {rho.extent_int(0), rho.extent_int(1), rho.extent_int(2)};
    int nfx = fx.extent_int(3);

    int const bc_idim = m_face.bc_idim;
    int const bc_iface = m_face.bc_iface;

    int const ng = m_grid->Nghost[bc_idim];
    if (bc_iface == 1) {
        begin[bc_idim] = rho.extent_int(bc_idim) - ng;
    }
    end[m_bc_idim] = begin[m_bc_idim] + ng;

    int const offset = m_bc_iface == 0 ? end[m_bc_idim] : begin[m_bc_idim] - 1;
    Kokkos::parallel_for(
            m_label,
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
            KOKKOS_LAMBDA(int i, int j, int k) {
                Kokkos::Array<int, 3> offsets {i, j, k};
                offsets[bc_idim] = offset;
                rho(i, j, k) = rho(offsets[0], offsets[1], offsets[2]);
                for (int n = 0; n < rhou.extent_int(3); ++n) {
                    rhou(i, j, k, n) = rhou(offsets[0], offsets[1], offsets[2], n);
                }
                E(i, j, k) = E(offsets[0], offsets[1], offsets[2]);
                for (int ifx = 0; ifx < nfx; ++ifx) {
                    fx(i, j, k, ifx) = fx(offsets[0], offsets[1], offsets[2], ifx);
                }
            });
}

PeriodicCondition::PeriodicCondition(int const idim, int const iface) noexcept
    : IBoundaryCondition(idim, iface)
{
}

void PeriodicCondition::execute(
        [[maybe_unused]] KV_double_3d const rho,
        [[maybe_unused]] KV_double_4d const rhou,
        [[maybe_unused]] KV_double_3d const E,
        [[maybe_unused]] KV_double_4d const fx) const noexcept
{
    // do nothing
}

ReflexiveCondition::ReflexiveCondition(int const idim, int const iface, Grid const& grid)
    : IBoundaryCondition(idim, iface)
    , m_label("Reflexive" + bc_dir[idim] + bc_face[iface])
    , m_grid(std::make_shared<Grid>(grid))
{
}

void ReflexiveCondition::execute(
        KV_double_3d const rho,
        KV_double_4d const rhou,
        KV_double_3d const E,
        KV_double_4d const fx) const noexcept
{
    Kokkos::Array<int, 3> begin {0, 0, 0};
    Kokkos::Array<int, 3> end {rho.extent_int(0), rho.extent_int(1), rho.extent_int(2)};
    int nfx = fx.extent_int(3);

    int const ng = m_grid->Nghost[m_bc_idim];
    if (m_bc_iface == 1) {
        begin[m_bc_idim] = rho.extent_int(m_bc_idim) - ng;
    }
    end[m_bc_idim] = begin[m_bc_idim] + ng;

    int const mirror = m_bc_iface == 0 ? (2 * ng - 1) : (2 * (rho.extent(m_bc_idim) - ng) - 1);
    int const& bc_idim = m_bc_idim;
    Kokkos::parallel_for(
            m_label,
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
            KOKKOS_LAMBDA(int i, int j, int k) {
                Kokkos::Array<int, 3> offsets {i, j, k};
                offsets[bc_idim] = mirror - offsets[bc_idim];
                rho(i, j, k) = rho(offsets[0], offsets[1], offsets[2]);
                for (int n = 0; n < rhou.extent_int(3); ++n) {
                    rhou(i, j, k, n) = rhou(offsets[0], offsets[1], offsets[2], n);
                }
                rhou(i, j, k, bc_idim) = -rhou(i, j, k, bc_idim);
                E(i, j, k) = E(offsets[0], offsets[1], offsets[2]);
                for (int ifx = 0; ifx < nfx; ++ifx) {
                    fx(i, j, k, ifx) = fx(offsets[0], offsets[1], offsets[2], ifx);
                }
            });
}

} // namespace novapp

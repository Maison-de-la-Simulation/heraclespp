//!
//! @file geometry.cpp
//!

#include <array>
#include <memory>
#include <stdexcept>

#include <Kokkos_Core.hpp>
#include <geom.hpp>
#include <kokkos_shortcut.hpp>
#include <ndim.hpp>

#include "geometry.hpp"

namespace novapp
{

IComputeGeom::IComputeGeom() = default;

IComputeGeom::IComputeGeom([[maybe_unused]] IComputeGeom const& rhs) = default;

IComputeGeom::IComputeGeom([[maybe_unused]] IComputeGeom&& rhs) noexcept = default;

IComputeGeom::~IComputeGeom() noexcept = default;

IComputeGeom& IComputeGeom::operator=([[maybe_unused]] IComputeGeom const& rhs) = default;

IComputeGeom& IComputeGeom::operator=([[maybe_unused]] IComputeGeom&& rhs) noexcept = default;

void Cartesian::execute(
    std::array<int, 3> const Nx_local_wg,
    [[maybe_unused]] std::array<int, 3> const Nghost,
    [[maybe_unused]] KV_cdouble_1d const x,
    [[maybe_unused]] KV_cdouble_1d const y,
    [[maybe_unused]] KV_cdouble_1d const z,
    KV_cdouble_1d const dx,
    KV_cdouble_1d const dy,
    KV_cdouble_1d const dz,
    KV_double_4d const ds,
    KV_double_3d const dv) const
{
    using policy_t = Kokkos::MDRangePolicy<Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>>;
    policy_t const policy({0, 0, 0}, {Nx_local_wg[0], Nx_local_wg[1], Nx_local_wg[2]});

    Kokkos::parallel_for(
        "fill_ds_cartesian",
        policy,
        KOKKOS_LAMBDA(int i, int j, int k)
        {
            Kokkos::Array<double, 3> dx_inter;
            for (int idim = 0; idim < 3; ++idim)
            {
                dx_inter[0] = dx(i);
                dx_inter[1] = dy(j);
                dx_inter[2] = dz(k);
                dx_inter[idim] = 1;
                ds(i, j, k, idim) = dx_inter[0] * dx_inter[1] * dx_inter[2];
            }
        });

    Kokkos::parallel_for(
        "fill_dv_cartesian",
        policy,
        KOKKOS_LAMBDA(int i, int j, int k)
        {
            dv(i, j, k) = dx(i) * dy(j) * dz(k);
        });
}

void Spherical::execute(
    std::array<int, 3> const Nx_local_wg,
    std::array<int, 3> const Nghost,
    KV_cdouble_1d const x,
    KV_cdouble_1d const y,
    [[maybe_unused]] KV_cdouble_1d const z,
    KV_cdouble_1d const dx,
    KV_cdouble_1d const dy,
    KV_cdouble_1d const dz,
    KV_double_4d const ds,
    KV_double_3d const dv) const
{
    using policy_t = Kokkos::MDRangePolicy<Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>>;
    policy_t const policy({0, 0, 0}, {Nx_local_wg[0], Nx_local_wg[1], Nx_local_wg[2]});

    if (ndim == 1)
    {
        // theta = pi
        // phi = 2 * pi
        Kokkos::parallel_for(
            "fill_ds_1dspherical",
            policy,
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                ds(i, j, k, 0) = 4 * Kokkos::numbers::pi * x(i) * x(i);
            });

        Kokkos::parallel_for(
            "fill_dv_1dspherical",
            policy,
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                dv(i, j, k) = (4 * Kokkos::numbers::pi * (x(i+1) * x(i+1) * x(i+1)
                            - (x(i) * x(i) * x(i)))) / 3;
            });
    }

    if (ndim == 2)
    {
        throw std::runtime_error("Spherical not available");
    }

    if (ndim == 3)
    {
        Kokkos::parallel_for(
            "fill_ds_3dspherical",
            policy,
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                double const dcos = Kokkos::cos(y(j)) - Kokkos::cos(y(j+1));

                ds(i, j, k, 0) = x(i) * x(i) * dcos * dz(k);              //r = cst
                ds(i, j, k, 1) = (1. / 2) * (x(i+1) * x(i+1) - (x(i) * x(i)))
                                * Kokkos::sin(y(j)) * dz(k); // theta = cst
                ds(i, j, k, 2) = x(i) * dx(i) * dy(j);                     //phi = cst
            });

        Kokkos::parallel_for(
            "fill_dv_3dspherical",
            policy,
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                double const dcos = Kokkos::cos(y(j)) - Kokkos::cos(y(j+1));
                dv(i, j, k) = (1. / 3) * (x(i+1) * x(i+1) * x(i+1)
                                - (x(i) * x(i) * x(i))) * dcos * dz(k);
            });
    }
}

} // namespace novapp

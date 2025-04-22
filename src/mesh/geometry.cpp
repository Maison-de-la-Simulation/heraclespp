// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file geometry.cpp
//!

#include <string>

#include <Kokkos_Core.hpp>
#include <kokkos_shortcut.hpp>
#include <ndim.hpp>

#include "geometry.hpp"
#include "range.hpp"

namespace novapp
{

IComputeGeom::IComputeGeom() = default;

IComputeGeom::IComputeGeom(IComputeGeom const& rhs) = default;

IComputeGeom::IComputeGeom(IComputeGeom&& rhs) noexcept = default;

IComputeGeom::~IComputeGeom() noexcept = default;

IComputeGeom& IComputeGeom::operator=(IComputeGeom const& /*rhs*/) = default;

IComputeGeom& IComputeGeom::operator=(IComputeGeom&& /*rhs*/) noexcept = default;

void Cartesian::execute(
    Range const& range,
    [[maybe_unused]] KV_cdouble_1d const& x,
    [[maybe_unused]] KV_cdouble_1d const& y,
    [[maybe_unused]] KV_cdouble_1d const& z,
    KV_cdouble_1d const& dx,
    KV_cdouble_1d const& dy,
    KV_cdouble_1d const& dz,
    KV_double_4d const& ds,
    KV_double_3d const& dv) const
{
    Kokkos::parallel_for(
        "fill_ds_cartesian",
        cell_mdrange(range),
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
        cell_mdrange(range),
        KOKKOS_LAMBDA(int i, int j, int k)
        {
            dv(i, j, k) = dx(i) * dy(j) * dz(k);
        });
}

void Spherical::execute(
    Range const& range,
    KV_cdouble_1d const& x,
    KV_cdouble_1d const& y,
    [[maybe_unused]] KV_cdouble_1d const& z,
    KV_cdouble_1d const& dx,
    KV_cdouble_1d const& dy,
    KV_cdouble_1d const& dz,
    KV_double_4d const& ds,
    KV_double_3d const& dv) const
{
    if (ndim == 1)
    {
        // theta = pi
        // phi = 2 * pi
        Kokkos::parallel_for(
            "fill_ds_1dspherical",
            cell_mdrange(range),
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                ds(i, j, k, 0) = 4 * Kokkos::numbers::pi * x(i) * x(i);
            });

        Kokkos::parallel_for(
            "fill_dv_1dspherical",
            cell_mdrange(range),
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                dv(i, j, k) = (4 * Kokkos::numbers::pi * (x(i+1) * x(i+1) * x(i+1)
                            - (x(i) * x(i) * x(i)))) / 3;
            });
    }

    if (ndim == 2)
    {
        // phi = 2 * pi
        Kokkos::parallel_for(
            "fill_ds_2dspherical",
            cell_mdrange(range),
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                double const dcos = Kokkos::cos(y(j)) - Kokkos::cos(y(j+1));
                ds(i, j, k, 0) = 2 * Kokkos::numbers::pi * x(i) * x(i) * dcos;              //r = cst
                ds(i, j, k, 1) = (2 * Kokkos::numbers::pi / 2) * (x(i+1) * x(i+1) - (x(i) * x(i)))
                                * Kokkos::sin(y(j)); // theta = cst
            });

        Kokkos::parallel_for(
            "fill_dv_2dspherical",
            cell_mdrange(range),
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                double const dcos = Kokkos::cos(y(j)) - Kokkos::cos(y(j+1));
                dv(i, j, k) = (2 * Kokkos::numbers::pi / 3) * (x(i+1) * x(i+1) * x(i+1)
                                - (x(i) * x(i) * x(i))) * dcos;
            });
    }

    if (ndim == 3)
    {
        Kokkos::parallel_for(
            "fill_ds_3dspherical",
            cell_mdrange(range),
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
            cell_mdrange(range),
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                double const dcos = Kokkos::cos(y(j)) - Kokkos::cos(y(j+1));
                dv(i, j, k) = (1. / 3) * (x(i+1) * x(i+1) * x(i+1)
                                - (x(i) * x(i) * x(i))) * dcos * dz(k);
            });
    }
}

} // namespace novapp

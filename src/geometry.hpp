//!
//! @file geometry.hpp
//!

#pragma once

#include "grid.hpp"
#include "ndim.hpp"
#include "geom.hpp"
#include "units.hpp"

namespace novapp
{

class IComputeGeom
{
public:
    IComputeGeom() = default;

    IComputeGeom(IComputeGeom const& rhs) = default;

    IComputeGeom(IComputeGeom&& rhs) noexcept = default;

    virtual ~IComputeGeom() noexcept = default;

    IComputeGeom& operator=(IComputeGeom const& rhs) = default;

    IComputeGeom& operator=(IComputeGeom&& rhs) noexcept = default;

    virtual void execute(
        KV_cdouble_1d x,
        KV_cdouble_1d y,
        KV_cdouble_1d dx,
        KV_cdouble_1d dy,
        KV_cdouble_1d dz,
        KV_double_4d ds,
        KV_double_3d dv,
        std::array<int, 3> Nx_local_wg) const
        = 0;
};

class Cartesian : public IComputeGeom
{
public:
    void execute(
        [[maybe_unused]] KV_cdouble_1d const x,
        [[maybe_unused]] KV_cdouble_1d const y,
        KV_cdouble_1d const dx,
        KV_cdouble_1d const dy,
        KV_cdouble_1d const dz,
        KV_double_4d ds,
        KV_double_3d dv,
        std::array<int, 3> Nx_local_wg) const final
    {
        KV_double_1d dx_inter("dx_inter", 3);

        Kokkos::parallel_for(
            "fill_ds_cartesian",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, 
                                {Nx_local_wg[0], 
                                 Nx_local_wg[1],
                                 Nx_local_wg[2]}),
            KOKKOS_CLASS_LAMBDA(int i, int j, int k)
            {
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
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, 
                                    {Nx_local_wg[0], 
                                     Nx_local_wg[1],
                                     Nx_local_wg[2]}),
            KOKKOS_CLASS_LAMBDA(int i, int j, int k)
            {
                dv(i, j, k) = dx(i) * dy(j) * dz(k);
            });
    }
};

class Spherical : public IComputeGeom
{
public:
    void execute(KV_cdouble_1d const x,
        KV_cdouble_1d const y,
        KV_cdouble_1d const dx,
        KV_cdouble_1d const dy,
        KV_cdouble_1d const dz,
        KV_double_4d ds,
        KV_double_3d dv,
        std::array<int, 3> Nx_local_wg) const final
    {
        if (ndim == 1)
        {
            // theta = pi
            // phi = 2 * pi
            Kokkos::parallel_for(
                "Fill_ds_1dspherical",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, 
                                    {Nx_local_wg[0], 
                                     Nx_local_wg[1],
                                     Nx_local_wg[2]}),
                KOKKOS_CLASS_LAMBDA(int i, int j, int k)
                {
                    ds(i, j, k, 0) = 4 * units::pi * x(i) * x(i);
                });

            Kokkos::parallel_for(
                "Fill_dv_1dspherical",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, 
                                        {Nx_local_wg[0], 
                                         Nx_local_wg[1],
                                         Nx_local_wg[2]}),
                KOKKOS_CLASS_LAMBDA(int i, int j, int k)
                {
                    dv(i, j, k) = (4 * units::pi * (x(i+1) * x(i+1) * x(i+1)
                                - (x(i) * x(i) * x(i)))) / 3;
                });
            dv(0, 0, 0) = dv(3, 0, 0);
            dv(1, 0, 0) = dv(2, 0, 0);
        }

        /* if (ndim == 3)
        {
            Kokkos::parallel_for(
            "Fill_ds_3dspherical",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, 
                                {Nx_local_wg[0], 
                                 Nx_local_wg[1],
                                 Nx_local_wg[2]}),
            KOKKOS_CLASS_LAMBDA(int i, int j, int k)
            {
                double dcost = Kokkos::cos(y(j)) - Kokkos::cos(y(j+1));

                ds(i, j, k, 0) = x(i) * x(i) * dcost * dz(k);              //r = cst
                ds(i, j, k, 1) = x(i) * Kokkos::sin(y(j)) * dx(i) * dz(k); // theta = cst
                ds(i, j, k, 2) = x(i) * dx(i) * dy(j);                     //phi = cst
            });

            Kokkos::parallel_for(
            "Fill_dv_3dspherical",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, 
                                    {Nx_local_wg[0], 
                                     Nx_local_wg[1],
                                     Nx_local_wg[2]}),
            KOKKOS_CLASS_LAMBDA(int i, int j, int k)
            {
                double dcost = Kokkos::cos(y(j)) - Kokkos::cos(y(j+1));
                dv(i, j, k) = (1 / 3) * (x(i+1) * x(i+1) * x(i+1)) 
                                - (x(i) * x(i) * x(i)) * dcost * dz(k);
            });
        }

        if (ndim == 2)
        {
            throw std::runtime_error("Spherical not available");
        } */
    }
};

inline std::unique_ptr<IComputeGeom> factory_grid_geometry()
{
    if (geom_choice == "CARTESIAN")
    {
        return std::make_unique<Cartesian>();
    }
    if (geom_choice == "SPHERICAL")
    {
        return std::make_unique<Spherical>();
    }
    throw std::runtime_error("Invalid grid geometry: .");
}

} // namespace novapp

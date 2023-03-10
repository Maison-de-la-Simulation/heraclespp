//!
//! @file gravity_implementation.hpp
//!
#pragma once

#include "ndim.hpp"

namespace novapp
{

class IGravity
{
public:
    IGravity() = default;

    IGravity(IGravity const& x) = default;

    IGravity(IGravity&& x) noexcept = default;

    virtual ~IGravity() noexcept = default;

    IGravity& operator=(IGravity const& x) = default;

    IGravity& operator=(IGravity&& x) noexcept = default;

    virtual void execute(
            KV_cdouble_3d rho,
            KV_cdouble_4d rhou,
            KV_double_4d rhou_new,
            KV_double_3d E_new,
            KV_double_1d g_array,
            double const dt) const
            = 0;
};

class GravityOn : public IGravity
{
public :
    virtual void execute(
            KV_cdouble_3d const rho,
            KV_cdouble_4d const rhou,
            KV_double_4d const rhou_new,
            KV_double_3d const E_new,
            KV_double_1d g_array,
            double const dt) const final
    {
        int istart = 2; // Default = 1D
        int jstart = 0;
        int kstart = 0;
        int iend = rho.extent(0) - 2;
        int jend = 1;
        int kend = 1;
            
        if (ndim == 2) // 2D
        {
            jstart = 2;
            jend = rho.extent(1) - 2;
        }
        if (ndim == 3) // 3D
        {
            jstart = 2;
            kstart = 2;
            jend = rho.extent(1) - 2;
            kend = rho.extent(2) - 2;
        }
        Kokkos::parallel_for(
        "GravityOn_implementation",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
        {istart, jstart, kstart},
        {iend, jend, kend}),
        KOKKOS_CLASS_LAMBDA(int i, int j, int k)
        {
            for (int idim = 0; idim < ndim; ++idim)
            {
                rhou_new(i, j, k, idim) += dt * g_array(idim) * rho(i, j, k);
                E_new(i, j, k) += dt * g_array(idim) * rhou(i, j, k, idim);
            }
        });
    }
};

class GravityOff : public IGravity
{
public :
    virtual void execute(
            [[maybe_unused]]KV_cdouble_3d const rho,
            [[maybe_unused]]KV_cdouble_4d const rhou,
            [[maybe_unused]]KV_double_4d const rhou_new,
            [[maybe_unused]]KV_double_3d const E_new,
            [[maybe_unused]]KV_double_1d g_array,
            [[maybe_unused]]double const dt) const final
    {
        // do nothing
    }
};

inline std::unique_ptr<IGravity> factory_gravity_source(
        std::string const& gravity)
{
    if (gravity == "On")
    {
        return std::make_unique<GravityOn>();
    }
    if (gravity == "Off")
    {
        return std::make_unique<GravityOff>();
    }
    throw std::runtime_error("Invalid gravity: " + gravity + ".");
}

} // namespace novapp

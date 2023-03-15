//!
//! @file godunov_scheme.hpp
//!

#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#include <PerfectGas.hpp>

#include "euler_equations.hpp"
#include "kronecker.hpp"
#include "ndim.hpp"
#include "range.hpp"
#include "riemann_solver.hpp"
#include "Kokkos_shortcut.hpp"

namespace novapp
{

class IGodunovScheme
{
public:
    IGodunovScheme() = default;

    IGodunovScheme(IGodunovScheme const& x) = default;

    IGodunovScheme(IGodunovScheme&& x) noexcept = default;

    virtual ~IGodunovScheme() noexcept = default;

    IGodunovScheme& operator=(IGodunovScheme const& x) = default;

    IGodunovScheme& operator=(IGodunovScheme&& x) noexcept = default;

    virtual void execute(
            Range const& range,
            KV_cdouble_3d rho,
            KV_cdouble_4d rhou,
            KV_cdouble_3d E,
            KV_cdouble_5d rho_rec,
            KV_cdouble_6d rhou_rec,
            KV_cdouble_5d E_rec,
            KV_double_3d rho_new,
            KV_double_4d rhou_new,
            KV_double_3d E_new,
            KV_cdouble_1d dx,
            double dt) const
            = 0;
};

template <class RiemannSolver>
class RiemannBasedGodunovScheme : public IGodunovScheme
{
    static_assert(
            std::is_invocable_r_v<
                    EulerFlux,
                    RiemannSolver,
                    EulerCons,
                    EulerCons,
                    int,
                    thermodynamics::PerfectGas>,
            "Incompatible Riemann solver.");

    RiemannSolver m_riemann_solver;

    thermodynamics::PerfectGas m_eos;
public:
    RiemannBasedGodunovScheme(
            RiemannSolver const& riemann_solver,
            thermodynamics::PerfectGas const& eos)
        : m_riemann_solver(riemann_solver)
        , m_eos(eos)
    {
    }

    void execute(
            Range const& range,
            KV_cdouble_3d const rho,
            KV_cdouble_4d const rhou,
            KV_cdouble_3d const E,
            KV_cdouble_5d const rho_rec,
            KV_cdouble_6d const rhou_rec,
            KV_cdouble_5d const E_rec,
            KV_double_3d const rho_new,
            KV_double_4d const rhou_new,
            KV_double_3d const E_new,
            KV_cdouble_1d const dx,
            double dt) const final
    {
        auto const [begin, end] = cell_range(range);
        Kokkos::parallel_for(
        "RiemannBasedGodunovScheme",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
        KOKKOS_CLASS_LAMBDA(int i, int j, int k) {
            rho_new(i, j, k) = rho(i, j, k);
            E_new(i, j, k) = E(i, j, k);
            for (int idim = 0; idim < ndim; ++idim)
            {
                rhou_new(i, j, k, idim) = rhou(i, j, k, idim);
            }
            for (int idim = 0; idim < ndim; ++idim)
            {
                int i_m = i - kron(idim, 0); // i - 1
                int i_p = i + kron(idim, 0); // i + 1
                int j_m = j - kron(idim, 1);
                int j_p = j + kron(idim, 1);
                int k_m = k - kron(idim, 2);
                int k_p = k + kron(idim, 2);

                EulerCons minus_oneR; // Right, back, top (i,j,k) - 1
                minus_oneR.density = rho_rec(i_m, j_m, k_m, 1, idim);
                for (int idr = 0; idr < ndim; ++idr)
                {
                    minus_oneR.momentum[idr] = rhou_rec(i_m, j_m, k_m, 1, idim, idr);
                }
                minus_oneR.energy = E_rec(i_m, j_m, k_m, 1, idim);
                EulerCons var_L; // Left, front, down (i,j,k)
                var_L.density = rho_rec(i, j, k, 0, idim);
                for (int idr = 0; idr < ndim; ++idr)
                {
                    var_L.momentum[idr] = rhou_rec(i, j, k, 0, idim, idr);
                }
                var_L.energy = E_rec(i, j, k, 0, idim);
                EulerFlux const FluxL = m_riemann_solver(minus_oneR, var_L, idim, m_eos);

                EulerCons var_R; // Right, back, top (i,j,k)
                var_R.density = rho_rec(i, j, k, 1, idim);
                for (int idr = 0; idr < ndim; ++idr)
                {
                    var_R.momentum[idr] = rhou_rec(i, j, k, 1, idim, idr);
                }
                var_R.energy = E_rec(i, j, k, 1, idim);
                EulerCons plus_oneL; // Left, front, down (i,j,k) + 1
                plus_oneL.density = rho_rec(i_p, j_p, k_p, 0, idim);
                for (int idr = 0; idr < ndim; ++idr)
                {
                    plus_oneL.momentum[idr] = rhou_rec(i_p, j_p, k_p, 0, idim, idr);
                }
                plus_oneL.energy = E_rec(i_p, j_p, k_p, 0, idim);
                EulerFlux const FluxR = m_riemann_solver(var_R, plus_oneL, idim, m_eos);

                double dtodx = dt / dx(idim);

                rho_new(i, j, k) += dtodx * (FluxL.density - FluxR.density);
                for (int idr = 0; idr < ndim; ++idr)
                {
                    rhou_new(i, j, k, idr) += dtodx * (FluxL.momentum[idr] - FluxR.momentum[idr]);
                }
                E_new(i, j, k) += dtodx * (FluxL.energy - FluxR.energy);

                // gravity ? interface
                /* rho_new(i, j, k) += dtodx * (FluxL.density - FluxR.density);
                for (int idr = 0; idr < ndim; ++idr)
                {
                    rhou_new(i, j, k, idr) += dtodx * (FluxL.momentum[idr] - FluxR.momentum[idr]);
                }
                rhou_new(i, j, k, idim) += - dt * kron(idim,1) * 0.1 * rho(i, j, k);
                E_new(i, j, k) += dtodx * (FluxL.energy - FluxR.energy) - dt * kron(idim,1) * 0.1 * rhou(i, j, k, idim); */
            }
        });
    }
};

inline std::unique_ptr<IGodunovScheme> factory_godunov_scheme(
        std::string const& riemann_solver,
        thermodynamics::PerfectGas const& eos)
{
    if (riemann_solver == "HLL")
    {
        return std::make_unique<RiemannBasedGodunovScheme<HLL>>(HLL(), eos);
    }
    throw std::runtime_error("Invalid riemann solver: " + riemann_solver + ".");
}

} // namespace novapp

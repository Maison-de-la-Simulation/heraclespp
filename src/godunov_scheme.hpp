//!
//! @file godunov_scheme.hpp
//!

#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#include "euler_equations.hpp"
#include "kronecker.hpp"
#include "ndim.hpp"
#include "range.hpp"
#include "grid.hpp"
#include "riemann_solver.hpp"
#include "kokkos_shortcut.hpp"
#include "gravity.hpp"
#include "nova_params.hpp"
#include "eos.hpp"

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
            Range const &range,
            double const dt,
            KV_cdouble_3d const rho,
            KV_cdouble_4d const rhou,
            KV_cdouble_3d const E,
            KV_cdouble_4d const fx,
            KV_cdouble_5d const rho_rec,
            KV_cdouble_6d const rhou_rec,
            KV_cdouble_5d const E_rec,
            KV_cdouble_6d const fx_rec,
            KV_double_3d rho_new,
            KV_double_4d rhou_new,
            KV_double_3d E_new,
            KV_double_4d const fx_new) const = 0;
    };

    template <class RiemannSolver, class Gravity>
    class RiemannBasedGodunovScheme : public IGodunovScheme
    {
        static_assert(
            std::is_invocable_r_v<
                    EulerFlux,
                    RiemannSolver,
                    EulerCons,
                    EulerCons,
                    int,
                    EOS>,
            "Incompatible Riemann solver.");

    private:
        RiemannSolver m_riemann_solver;
        Gravity m_gravity;
        EOS m_eos;
        Grid m_grid;
        
    public:
        RiemannBasedGodunovScheme(
            RiemannSolver const &riemann_solver,
            Gravity const& gravity,
            EOS const& eos, 
            Grid const& grid)
            : m_riemann_solver(riemann_solver)
            , m_gravity(gravity)
            , m_eos(eos)
            , m_grid(grid)
        {
        }

        void execute(
            Range const& range,
            double const dt,
            KV_cdouble_3d const rho,
            KV_cdouble_4d const rhou,
            KV_cdouble_3d const E,
            KV_cdouble_4d const fx,
            KV_cdouble_5d const rho_rec,
            KV_cdouble_6d const rhou_rec,
            KV_cdouble_5d const E_rec,
            KV_cdouble_6d const fx_rec,
            KV_double_3d const rho_new,
            KV_double_4d const rhou_new,
            KV_double_3d const E_new,
            KV_double_4d const fx_new) const final
        {
            auto const [begin, end] = cell_range(range);
            my_parallel_for(begin, end,
            KOKKOS_CLASS_LAMBDA(int i, int j, int k) KOKKOS_IMPL_HOST_FORCEINLINE
            {
                double dsodv_alloc[2*ndim];
                Kokkos::View<double[2][ndim]> dsodv(dsodv_alloc);

                double const invdv = 1 / m_grid.dv(i, j, k);

                NOVA_FORCEUNROLL
                for (int idim = 0; idim < ndim; ++idim) {
                    auto const [i_p, j_p, k_p] = rindex(idim, i, j, k); // i + 1
                    dsodv(0, idim) = m_grid.ds(i, j, k, idim) * invdv;
                    dsodv(1, idim) = m_grid.ds(i_p, j_p, k_p, idim) * invdv;
                }

                EulerCons cons_loc;
                cons_loc.rho = rho(i, j, k);
                cons_loc.E = E(i, j, k);
                NOVA_FORCEUNROLL
                for (int idim = 0; idim < ndim; ++idim)
                {
                    cons_loc.rhou[idim] = rhou(i, j, k, idim);
                }

                double fx_loc_alloc[nfx];
                Kokkos::View<double[nfx]> fx_loc(fx_loc_alloc);
                NOVA_FORCEUNROLL
                for (int ifx = 0; ifx < nfx; ++ifx)
                {
                    fx_loc(ifx) = fx(i, j, k, ifx) * cons_loc.rho;
                }

                NOVA_FORCEUNROLL
                for (int idim = 0; idim < ndim; ++idim)
                {
                    auto const [i_m, j_m, k_m] = lindex(idim, i, j, k); // i - 1
                    auto const [i_p, j_p, k_p] = rindex(idim, i, j, k); // i + 1

                    EulerCons minus_oneR; // Right, back, top (i,j,k) - 1
                    minus_oneR.rho = rho_rec(i_m, j_m, k_m, 1, idim);
                    NOVA_FORCEUNROLL
                    for (int idr = 0; idr < ndim; ++idr)
                    {
                        minus_oneR.rhou[idr] = rhou_rec(i_m, j_m, k_m, 1, idim, idr);
                    }
                    minus_oneR.E = E_rec(i_m, j_m, k_m, 1, idim);
                    EulerCons var_L; // Left, front, down (i,j,k)
                    var_L.rho = rho_rec(i, j, k, 0, idim);
                    NOVA_FORCEUNROLL
                    for (int idr = 0; idr < ndim; ++idr)
                    {
                        var_L.rhou[idr] = rhou_rec(i, j, k, 0, idim, idr);
                    }
                    var_L.E = E_rec(i, j, k, 0, idim);
                    EulerFlux const FluxL = m_riemann_solver(minus_oneR, var_L, idim, m_eos);

                    EulerCons var_R; // Right, back, top (i,j,k)
                    var_R.rho = rho_rec(i, j, k, 1, idim);
                    NOVA_FORCEUNROLL
                    for (int idr = 0; idr < ndim; ++idr)
                    {
                        var_R.rhou[idr] = rhou_rec(i, j, k, 1, idim, idr);
                    }
                    var_R.E = E_rec(i, j, k, 1, idim);
                    EulerCons plus_oneL; // Left, front, down (i,j,k) + 1
                    plus_oneL.rho = rho_rec(i_p, j_p, k_p, 0, idim);
                    NOVA_FORCEUNROLL
                    for (int idr = 0; idr < ndim; ++idr)
                    {
                        plus_oneL.rhou[idr] = rhou_rec(i_p, j_p, k_p, 0, idim, idr);
                    }
                    plus_oneL.E = E_rec(i_p, j_p, k_p, 0, idim);
                    EulerFlux const FluxR = m_riemann_solver(var_R, plus_oneL, idim, m_eos);

                    cons_loc.rho += dt * (FluxL.rho * dsodv(0, idim) - FluxR.rho * dsodv(1, idim));
                    NOVA_FORCEUNROLL
                    for (int idr = 0; idr < ndim; ++idr)
                    {
                        cons_loc.rhou[idr] += dt * (FluxL.rhou[idr] * dsodv(0, idim) - FluxR.rhou[idr] * dsodv(1, idim));
                    }
                    cons_loc.E += dt * (FluxL.E * dsodv(0, idim) - FluxR.E * dsodv(1, idim));

                    // Gravity
                    cons_loc.rhou[idim] += dt * m_gravity(i, j, k, idim, m_grid) * rho(i, j, k);
                    cons_loc.E += dt * m_gravity(i, j, k, idim, m_grid) * rhou(i, j, k, idim);

                    // Passive scalar
                    NOVA_FORCEUNROLL
                    for (int ifx = 0; ifx < nfx; ++ifx)
                    {
                        int iL_uw = i_m; // upwind
                        int iR_uw = i;
                        int jL_uw = j_m;
                        int jR_uw = j;
                        int kL_uw = k_m;
                        int kR_uw = k;
                        int face_L = 1;
                        int face_R = 1;

                        if (FluxL.rho < 0)
                        {
                            iL_uw = i;
                            jL_uw = j;
                            kL_uw = k;
                            face_L = 0;
                        }
                        if (FluxR.rho < 0)
                        {
                            iR_uw = i_p;
                            jR_uw = j_p;
                            kR_uw = k_p;
                            face_R = 0;
                        }

                        double const flux_fx_L = fx_rec(iL_uw, jL_uw, kL_uw, face_L, idim, ifx) * FluxL.rho;
                        double const flux_fx_R = fx_rec(iR_uw, jR_uw, kR_uw, face_R, idim, ifx) * FluxR.rho;

                        fx_loc(ifx) += dt * (flux_fx_L * dsodv(0, idim) - flux_fx_R * dsodv(1, idim));
                    }
                }

                rho_new(i, j, k) = cons_loc.rho;
                E_new(i, j, k) = cons_loc.E;
                NOVA_FORCEUNROLL
                for (int idim = 0; idim < ndim; ++idim)
                {
                    rhou_new(i, j, k, idim) = cons_loc.rhou[idim];
                }

                NOVA_FORCEUNROLL
                for (int ifx = 0; ifx < nfx; ++ifx)
                {
                    double fx = fx_loc(ifx) / cons_loc.rho;
                    fx = Kokkos::fmax(fx, 0.);
                    fx = Kokkos::fmin(fx, 1.);
                    fx_new(i, j, k, ifx) = fx;
                }
            });
        }
    };

    inline std::unique_ptr<IGodunovScheme> factory_godunov_scheme(
        std::string const& riemann_solver,
        std::string const& gravity,
        EOS const& eos, 
        Grid const& grid,
        KV_double_1d const& g)
    {
        if (riemann_solver == "HLL" && gravity == "Uniform")
        {
            return std::make_unique<RiemannBasedGodunovScheme<HLL, UniformGravity>>(HLL(), UniformGravity(g), eos, grid);
        }
        throw std::runtime_error("Invalid riemann solver: " + riemann_solver + "or gravity: " + gravity + ".");
    }

} // namespace novapp

// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file face_reconstruction.hpp
//!

#pragma once

#include <cassert>

#include <Kokkos_Core.hpp>
#include <grid.hpp>
#include <kokkos_shortcut.hpp>
#include <kronecker.hpp>
#include <ndim.hpp>
#include <range.hpp>

#include "concepts.hpp"
#include "face_reconstruction.hpp"

namespace novapp
{

template <concepts::SlopeLimiter SlopeLimiter>
class LimitedLinearReconstruction : public IFaceReconstruction
{
private:
    SlopeLimiter m_slope_limiter;

public:
    explicit LimitedLinearReconstruction(SlopeLimiter const& slope_limiter)
        : m_slope_limiter(slope_limiter)
    {
    }

    void execute(
        Range const& range,
        Grid const& grid,
        KV_cdouble_3d const& var,
        KV_double_5d const& var_rec) const final
    {
        assert(equal_extents({0, 1, 2}, var, var_rec));
        assert(var_rec.extent(3) == 2);
        assert(var_rec.extent(4) == ndim);

        auto const& slope_limiter = m_slope_limiter;

        KV_cdouble_1d const dx = grid.dx;
        KV_cdouble_1d const dy = grid.dy;
        KV_cdouble_1d const dz = grid.dz;

        Kokkos::parallel_for(
            "face_reconstruction",
            cell_mdrange(range),
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                for (int idim = 0; idim < ndim; ++idim)
                {
                    auto const [i_m, j_m, k_m] = lindex(idim, i, j, k); // i - 1
                    auto const [i_p, j_p, k_p] = rindex(idim, i, j, k); // i + 1
                    double const dl = (kron(idim,0) * dx(i))
                                    + (kron(idim,1) * dy(j))
                                    + (kron(idim,2) * dz(k));
                    double const dl_m = (kron(idim,0) * dx(i_m))
                                      + (kron(idim,1) * dy(j_m))
                                      + (kron(idim,2) * dz(k_m));
                    double const dl_p = (kron(idim,0) * dx(i_p))
                                      + (kron(idim,1) * dy(j_p))
                                      + (kron(idim,2) * dz(k_p));

                    double const slope = slope_limiter(
                        (var(i_p, j_p, k_p) - var(i, j, k)) / ((dl + dl_p) / 2),
                        (var(i, j, k) - var(i_m, j_m, k_m)) / ((dl_m + dl) / 2));

                    var_rec(i, j, k, 0, idim) =  var(i, j, k) - ((dl / 2) * slope);
                    var_rec(i, j, k, 1, idim) =  var(i, j, k) + ((dl / 2) * slope);
                }
            });
    }
};

} // namespace novapp

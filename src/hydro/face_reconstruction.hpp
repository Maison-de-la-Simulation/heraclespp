//!
//! @file face_reconstruction.hpp
//!

#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <Kokkos_Core.hpp>
#include <grid.hpp>
#include <kokkos_shortcut.hpp>
#include <kronecker.hpp>
#include <ndim.hpp>
#include <range.hpp>

#include "slope_limiters.hpp"

namespace novapp
{

class IFaceReconstruction
{
public:
    IFaceReconstruction() = default;

    IFaceReconstruction(IFaceReconstruction const& rhs) = default;

    IFaceReconstruction(IFaceReconstruction&& rhs) noexcept = default;

    virtual ~IFaceReconstruction() noexcept = default;

    IFaceReconstruction& operator=(IFaceReconstruction const& rhs) = default;

    IFaceReconstruction& operator=(IFaceReconstruction&& rhs) noexcept = default;

    //! @param[in] range output iteration range
    //! @param[in] var cell values
    //! @param[out] var_rec reconstructed values at interfaces
    virtual void execute(
        Range const& range,
        KV_cdouble_3d const& var,
        KV_double_5d const& var_rec) const
        = 0;
};

template <class SlopeLimiter>
class LimitedLinearReconstruction : public IFaceReconstruction
{
    static_assert(
            std::is_invocable_r_v<
            double,
            SlopeLimiter,
            double,
            double>,
            "Invalid slope limiter.");

private:
    SlopeLimiter m_slope_limiter;
    Grid m_grid;

public:
    LimitedLinearReconstruction(
            SlopeLimiter const& slope_limiter,
            Grid const& grid)
        : m_slope_limiter(slope_limiter)
        , m_grid(grid)
    {
    }

    void execute(
        Range const& range,
        KV_cdouble_3d const& var,
        KV_double_5d const& var_rec) const final
    {
        assert(var.extent(0) == var_rec.extent(0));
        assert(var.extent(1) == var_rec.extent(1));
        assert(var.extent(2) == var_rec.extent(2));

        auto const& grid = m_grid;
        auto const& slope_limiter = m_slope_limiter;

        Kokkos::parallel_for(
            "face_reconstruction",
            cell_mdrange(range),
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                for (int idim = 0; idim < ndim; ++idim)
                {
                    auto const [i_m, j_m, k_m] = lindex(idim, i, j, k); // i - 1
                    auto const [i_p, j_p, k_p] = rindex(idim, i, j, k); // i + 1
                    double const dx = kron(idim,0) * grid.dx(i)
                                    + kron(idim,1) * grid.dy(j)
                                    + kron(idim,2) * grid.dz(k);
                    double const dx_m = kron(idim,0) * grid.dx(i_m)
                                      + kron(idim,1) * grid.dy(j_m)
                                      + kron(idim,2) * grid.dz(k_m);
                    double const dx_p = kron(idim,0) * grid.dx(i_p)
                                      + kron(idim,1) * grid.dy(j_p)
                                      + kron(idim,2) * grid.dz(k_p);

                    double const slope = slope_limiter(
                        (var(i_p, j_p, k_p) - var(i, j, k)) / ((dx + dx_p) / 2),
                        (var(i, j, k) - var(i_m, j_m, k_m)) / ((dx_m + dx) / 2));

                    var_rec(i, j, k, 0, idim) =  var(i, j, k) - (dx / 2) * slope;
                    var_rec(i, j, k, 1, idim) =  var(i, j, k) + (dx / 2) * slope;
                }
            });
    }
};

inline std::unique_ptr<IFaceReconstruction> factory_face_reconstruction(
        std::string const& slope,
        Grid const& grid)
{
    if (slope == "Constant")
    {
        return std::make_unique<LimitedLinearReconstruction<Constant>>(Constant(), grid);
    }

    if (slope == "VanLeer")
    {
        return std::make_unique<LimitedLinearReconstruction<VanLeer>>(VanLeer(), grid);
    }

    if (slope == "Minmod")
    {
        return std::make_unique<LimitedLinearReconstruction<Minmod>>(Minmod(), grid);
    }

    if (slope == "VanAlbada")
    {
        return std::make_unique<LimitedLinearReconstruction<VanAlbada>>(VanAlbada(), grid);
    }

    throw std::runtime_error("Unknown face reconstruction algorithm: " + slope + ".");
}

} // namespace novapp

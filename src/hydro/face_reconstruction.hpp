//!
//! @file face_reconstruction.hpp
//!

#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <Kokkos_Core.hpp>

#include "../mesh/grid.hpp"
#include "kokkos_shortcut.hpp"
#include "kronecker.hpp"
#include "ndim.hpp"
#include "../mesh/range.hpp"
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

    //! @param[in] var cell values
    //! @param[out] varL left edge reconstruction values
    //! @param[out] varR right edge reconstruction values
    virtual void execute(
        Range const& range,
        KV_cdouble_3d var,
        KV_double_5d var_rec) const
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
        KV_cdouble_3d var,
        KV_double_5d var_rec) const final
    {
        assert(var.extent(0) == var_rec.extent(0));
        assert(var.extent(1) == var_rec.extent(1));
        assert(var.extent(2) == var_rec.extent(2));

        auto const [begin, end] = cell_range(range);
        Kokkos::parallel_for(
            "face_reconstruction",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
            KOKKOS_CLASS_LAMBDA(int i, int j, int k) 
            {
                for (int idim = 0; idim < ndim; ++idim)
                {
                    auto const [i_m, j_m, k_m] = lindex(idim, i, j, k); // i - 1
                    auto const [i_p, j_p, k_p] = rindex(idim, i, j, k); // i + 1
                    double dx = kron(idim,0) * m_grid.dx(i) 
                                + kron(idim,1) * m_grid.dy(j) 
                                + kron(idim,2) * m_grid.dz(k);
                    
                    double const slope = m_slope_limiter(
                        (var(i_p, j_p, k_p) - var(i, j, k)) / dx,
                        (var(i, j, k) - var(i_m, j_m, k_m)) / dx);

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

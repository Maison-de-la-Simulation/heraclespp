//!
//! @file face_reconstruction.hpp
//!

#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include "kronecker.hpp"
#include "ndim.hpp"
#include "range.hpp"
#include "slope_limiters.hpp"
#include "Kokkos_shortcut.hpp"


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
        KV_double_5d var_rec,
        Grid const& grid) const
        = 0;
};

template <class SlopeLimiter>
class LimitedLinearReconstruction : public IFaceReconstruction
{
    static_assert(
            std::is_invocable_r_v<double, SlopeLimiter, double, double>,
            "Invalid slope limiter.");

private:
    SlopeLimiter m_slope_limiter;

public:
    LimitedLinearReconstruction(SlopeLimiter const& slope_limiter)
        : m_slope_limiter(slope_limiter)
    {
    }

    void execute(
        Range const& range,
        KV_cdouble_3d var,
        KV_double_5d var_rec,
        Grid const& grid) const final
    {
        assert(var.extent(0) == var_rec.extent(0));
        assert(var.extent(1) == var_rec.extent(1));
        assert(var.extent(2) == var_rec.extent(2));
        
        auto const [begin, end] = cell_range(range);
        Kokkos::parallel_for(
        "LimitedLinearFaceReconstruction",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
        KOKKOS_CLASS_LAMBDA(int i, int j, int k) 
        {
            for (int idim = 0; idim < ndim; ++idim)
            {
                auto const [i_m, j_m, k_m] = lindex(idim, i, j, k); // i - 1
                auto const [i_p, j_p, k_p] = rindex(idim, i, j, k); // i + 1

                double const slope = m_slope_limiter(
                    (var(i_p, j_p, k_p) - var(i, j, k)) / grid.dx[idim],
                    (var(i, j, k) - var(i_m, j_m, k_m)) / grid.dx[idim]);

                var_rec(i, j, k, 0, idim) =  var(i, j, k) - (grid.dx[idim] / 2) * slope;
                var_rec(i, j, k, 1, idim) =  var(i, j, k) + (grid.dx[idim] / 2) * slope;
            } 
        });
    }
};

inline std::unique_ptr<IFaceReconstruction> factory_face_reconstruction(
        std::string const& label)
{
    if (label == Constant::s_label)
    {
        return std::make_unique<LimitedLinearReconstruction<Constant>>(Constant());
    }

    if (label == VanLeer::s_label)
    {
        return std::make_unique<LimitedLinearReconstruction<VanLeer>>(VanLeer());
    }

    if (label == Minmod::s_label)
    {
        return std::make_unique<LimitedLinearReconstruction<Minmod>>(Minmod());
    }

    if (label == VanAlbada::s_label)
    {
        return std::make_unique<LimitedLinearReconstruction<VanAlbada>>(VanAlbada());
    }

    throw std::runtime_error("Unknown face reconstruction algorithm: " + label + ".");
}

} // namespace novapp
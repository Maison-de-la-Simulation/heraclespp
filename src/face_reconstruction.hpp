//!
//! @file face_reconstruction.hpp
//!
#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include <Kokkos_Core.hpp>

#include "slope_limiters.hpp"
#include "kronecker.hpp"
#include "ndim.hpp"

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
        Kokkos::View<const double***, Kokkos::LayoutStride> var,
        Kokkos::View<double*****, Kokkos::LayoutStride> var_rec,
        Kokkos::View<const double*> dx) const
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
        Kokkos::View<const double***, Kokkos::LayoutStride> var,
        Kokkos::View<double*****, Kokkos::LayoutStride> var_rec,
        Kokkos::View<const double*> dx) const final
    {
        assert(var.extent(0) == var_rec.extent(0));
        assert(var.extent(1) == var_rec.extent(1));
        assert(var.extent(2) == var_rec.extent(2));
        {   
            int istart = 1; // Default = 1D
            int jstart = 0;
            int kstart = 0;
            int iend = var.extent(0) - 1;
            int jend = 1;
            int kend = 1;
            
            if (ndim == 2) // 2D
            {
                jstart = 1;
                jend = var.extent(1) - 1;
            }
            if (ndim == 3) // 3D
            {
                jstart = 1;
                kstart = 1;
                jend = var.extent(1) - 1;
                kend = var.extent(2) - 1;
            }
            
            Kokkos::parallel_for(
            "LimitedLinearFaceReconstruction",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {istart, jstart, kstart},
            {iend, jend, kend}),
            KOKKOS_CLASS_LAMBDA(int i, int j, int k) 
            {
                for (int idim = 0; idim < ndim; ++idim)
                {
                    int i_m = i - kron(idim, 0); // i - 1
                    int i_p = i + kron(idim, 0); // i + 1
                    int j_m = j - kron(idim, 1);
                    int j_p = j + kron(idim, 1);
                    int k_m = k - kron(idim, 2);
                    int k_p = k + kron(idim, 2);

                    double const slope = m_slope_limiter(
                        (var(i_p, j_p, k_p) - var(i, j, k)) / dx(idim),
                        (var(i, j, k) - var(i_m, j_m, k_m)) / dx(idim));

                    var_rec(i, j, k, 0, idim) =  var(i, j, k) - (dx(idim) / 2) * slope;
                    var_rec(i, j, k, 1, idim) =  var(i, j, k) + (dx(idim) / 2) * slope;
                } 
            });
        } 
    }
};

inline std::unique_ptr<IFaceReconstruction> factory_face_reconstruction(
        std::string const& label)
{
    if (label == Constant::s_label)
    {
        //return std::make_unique<ConstantReconstruction>();
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

    throw std::runtime_error("Unknown reconstruction algorithm: " + label + ".");
}
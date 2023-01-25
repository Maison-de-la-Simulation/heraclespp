//!
//! @file face_reconstruction.hpp
//!
#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include <Kokkos_Core.hpp>

#include "slope_limiters.hpp"

class IFaceReconstruction
{
public:
    IFaceReconstruction() = default;

    IFaceReconstruction(IFaceReconstruction const& x) = default;

    IFaceReconstruction(IFaceReconstruction&& x) noexcept = default;

    virtual ~IFaceReconstruction() noexcept = default;

    IFaceReconstruction& operator=(IFaceReconstruction const& x) = default;

    IFaceReconstruction& operator=(IFaceReconstruction&& x) noexcept = default;

    //! @param[in] var cell values
    //! @param[out] varL left edge reconstruction values
    //! @param[out] varR right edge reconstruction values
    virtual void execute(
            Kokkos::View<const double***> var,
            Kokkos::View<double***> varL,
            Kokkos::View<double***> varR) const = 0;
};

template <class SlopeLimiter>
class LimitedLinearReconstruction : public IFaceReconstruction
{
    static_assert(
            std::is_invocable_r_v<double, SlopeLimiter, double, double>,
            "Invalid slope limiter.");

private:
    SlopeLimiter m_slope_limiter;

    double m_dx;
    double m_half_dx;

public:
    LimitedLinearReconstruction(SlopeLimiter const& slope_limiter, double const dx)
        : m_slope_limiter(slope_limiter)
        , m_dx(dx)
        , m_half_dx(dx / 2)
    {
    }

    void execute(
            Kokkos::View<const double***> const var,
            Kokkos::View<double***> const varL,
            Kokkos::View<double***> const varR) const final
    {
        assert(var.extent(0) == varL.extent(0));
        assert(varL.extent(0) == varR.extent(0));
        assert(var.extent(1) == varL.extent(1));
        assert(varL.extent(1) == varR.extent(1));
        assert(var.extent(2) == varL.extent(2));
        assert(varL.extent(2) == varR.extent(2));
        Kokkos::parallel_for(
                "LimitedLinearFaceReconstruction",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                {1, 0, 0},
                {static_cast<int>(var.extent(0) - 1), static_cast<int>(var.extent(1)), static_cast<int>(var.extent(2))}), // Warning on (y,z) -> -1
                KOKKOS_LAMBDA(int i, int j, int k)
                {
                    double const slope_var = m_slope_limiter((var(i+1, j, k) - var(i, j, k)) / m_dx, (var(i, j, k) - var(i-1, j, k)) / m_dx);
                    varL(i, j, k) = var(i, j, k) - m_half_dx * slope_var;
                    varR(i, j, k) = var(i, j, k) + m_half_dx * slope_var;
                });
    }
};

class ConstantReconstruction : public IFaceReconstruction
{
public:
    void execute(
            Kokkos::View<const double***> const var,
            Kokkos::View<double***> const varL,
            Kokkos::View<double***> const varR) const final
    {
        assert(var.extent(0) == varL.extent(0));
        assert(varL.extent(0) == varR.extent(0));
        assert(var.extent(1) == varL.extent(1));
        assert(varL.extent(1) == varR.extent(1));
        assert(var.extent(2) == varL.extent(2));
        assert(varL.extent(2) == varR.extent(2));
        
        Kokkos::parallel_for(
                "ConstantReconstruction",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                {1, 0, 0},
                {static_cast<int>(var.extent(0) - 1), static_cast<int>(var.extent(1)), static_cast<int>(var.extent(2))}),
                KOKKOS_LAMBDA(int i, int j, int k) 
                {
                    varL(i, j, k) = var(i, j, k);
                    varR(i, j, k) = var(i, j, k);
                });
    }
};

inline std::unique_ptr<IFaceReconstruction> factory_face_reconstruction(
        std::string const& s,
        double dx)
{
    if (s == "Constant")
    {
        return std::make_unique<ConstantReconstruction>();
    }
    else if (s == VanLeer::s_label)
    {
        return std::make_unique<LimitedLinearReconstruction<VanLeer>>(VanLeer(), dx);
    }
    else if (s == Minmod::s_label)
    {
        return std::make_unique<LimitedLinearReconstruction<Minmod>>(Minmod(), dx);
    }
    else if (s == VanAlbada::s_label)
    {
        return std::make_unique<LimitedLinearReconstruction<VanAlbada>>(VanAlbada(), dx);
    }
    else
    {
        throw std::runtime_error("Unknown reconstruction algorithm: " + s + ".");
    }
}

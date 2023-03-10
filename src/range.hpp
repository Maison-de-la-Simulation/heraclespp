
/**
 * @file range.hpp
 * Grid class declaration
 */
#pragma once

#include <array>

#include <Kokkos_Core.hpp>

#include "ndim.hpp"

namespace novapp
{

class Range
{
public :

    std::array<int, 3> Corner_min ; // Bottom, left, back corner without ghost
    std::array<int, 3> Corner_max ; // top, right, front corner without ghost

    int Ng;
    int NgEff;
    std::array<int, 3> Nghost;

    std::array<int, 3> Nc_min_2g, Nc_max_2g ; // min, max of cell centered viarables with 2 ghost cells
    std::array<int, 3> Nf_min_2g, Nf_max_2g ; // min, max of face centered viarables with 2 ghost cells

    std::array<int, 3> Nc_min_1g, Nc_max_1g ; // min, max of cell centered viarables with 1 ghost cells
    std::array<int, 3> Nf_min_1g, Nf_max_1g ; // min, max of face centered viarables with 1 ghost cells

    std::array<int, 3> Nc_min_0g, Nc_max_0g ; // min, max of cell centered viarables with 1 ghost cells
    std::array<int, 3> Nf_min_0g, Nf_max_0g ; // min, max of face centered viarables with 1 ghost cells

    Range() = default;

    Range(std::array<int, 3> const& Cmin,
          std::array<int, 3> const& Cmax,
          int Nghost);

    Range no_ghosts() const noexcept
    {
        return with_ghosts(0);
    }

    Range all_ghosts() const noexcept
    {
        return with_ghosts(Ng);
    }

    Range with_ghosts(int const NgEff) const noexcept
    {
        Range rng(*this);
        rng.NgEff = NgEff;
        return rng;
    }
};

inline std::array<Kokkos::Array<int, 3>, 2> cell_range(Range const& range)
{
    Kokkos::Array<int, 3> begin;
    Kokkos::Array<int, 3> end;
    for (int idim = 0; idim < ndim; ++idim)
    {
        begin[idim] = range.Nghost[idim] - range.NgEff;
        end[idim] = range.Nghost[idim] + range.Corner_max[idim] - range.Corner_min[idim] + range.NgEff;
    }

    for (int idim = ndim; idim < 3; ++idim)
    {
        begin[idim] = range.Nghost[idim];
        end[idim] = range.Nghost[idim] + range.Corner_max[idim] - range.Corner_min[idim];
    }

    return std::array<Kokkos::Array<int, 3>, 2> {begin, end};
}

} // namespace novapp

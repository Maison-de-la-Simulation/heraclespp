//!
//! @file range.hpp
//! Grid class declaration
//!

#pragma once

#include <array>
#include <iosfwd>

#include <Kokkos_Core.hpp>

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

    Range();

    Range(std::array<int, 3> const& Cmin,
          std::array<int, 3> const& Cmax,
          int Nghost);

    Range no_ghosts() const;

    Range all_ghosts() const;

    Range with_ghosts(int NgEff) const;
};

std::ostream& operator<<(std::ostream& os, Range const& rng);

std::array<Kokkos::Array<int, 3>, 2> cell_range(Range const& range);

Kokkos::MDRangePolicy<Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>> cell_mdrange(
        Range const& range);

} // namespace novapp

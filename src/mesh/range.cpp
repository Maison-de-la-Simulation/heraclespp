// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

/**
 * @file range.cpp
 * Geom class implementation
 */

#include <array>
#include <iostream>
#include <stdexcept>

#include <Kokkos_Core.hpp>
#include <ndim.hpp>

#include "range.hpp"

namespace novapp {

Range::Range() = default;

Range::Range(std::array<int, 3> const& Cmin, std::array<int, 3> const& Cmax, int Ng)
    : Corner_min(Cmin)
    , Corner_max(Cmax)
    , Ng(Ng)
    , NgEff(0)
    , Nghost({0, 0, 0})
{
    for (int idim = 0; idim < ndim; ++idim) {
        Nghost[idim] = Ng;
    }

    for (int idim = 0; idim < 3; ++idim) {
        if (Cmin[idim] < 0 || Cmax[idim] < 0) {
            throw std::runtime_error("Cmin < 0 || Cmax < 0");
        }

        if (Cmin[idim] > Cmax[idim]) {
            throw std::runtime_error("Cmin > Cmax");
        }
    }

    for (int idim = 0; idim < ndim; ++idim) {
        // if (Nghost[idim] < 2)
        // {
        //     throw std::runtime_error("Nghost < 2");
        // }

        Nc_min_0g[idim] = Nghost[idim];
        Nc_max_0g[idim] = Nghost[idim] + Cmax[idim] - Cmin[idim];
        Nf_min_0g[idim] = Nc_min_0g[idim];
        Nf_max_0g[idim] = Nc_max_0g[idim] + 1;

        Nc_min_1g[idim] = Nc_min_0g[idim] - 1;
        Nc_max_1g[idim] = Nc_max_0g[idim] + 1;
        Nf_min_1g[idim] = Nc_min_1g[idim];
        Nf_max_1g[idim] = Nc_max_1g[idim] + 1;

        Nc_min_2g[idim] = Nc_min_0g[idim] - 2;
        Nc_max_2g[idim] = Nc_max_0g[idim] + 2;
        Nf_min_2g[idim] = Nc_min_2g[idim];
        Nf_max_2g[idim] = Nc_max_2g[idim] + 1;
    }

    for (int idim = ndim; idim < 3; ++idim) {
        Nc_min_0g[idim] = Nghost[idim];
        Nc_max_0g[idim] = Nghost[idim] + Cmax[idim] - Cmin[idim];
        Nf_min_0g[idim] = Nc_min_0g[idim];
        Nf_max_0g[idim] = Nc_max_0g[idim] + 1;

        Nc_min_1g[idim] = Nc_min_0g[idim];
        Nc_max_1g[idim] = Nc_max_0g[idim];
        Nf_min_1g[idim] = Nf_min_0g[idim];
        Nf_max_1g[idim] = Nf_max_0g[idim];

        Nc_min_2g[idim] = Nc_min_0g[idim];
        Nc_max_2g[idim] = Nc_max_0g[idim];
        Nf_min_2g[idim] = Nf_min_0g[idim];
        Nf_max_2g[idim] = Nf_max_0g[idim];
    }
}

Range::Range(std::array<int, 2> const& rng_x0, std::array<int, 2> const& rng_x1, std::array<int, 2> const& rng_x2, int const Nghost)
    : Range({rng_x0[0], rng_x1[0], rng_x2[0]}, {rng_x0[1], rng_x1[1], rng_x2[1]}, Nghost)
{
}

Range Range::no_ghosts() const
{
    return with_ghosts(0);
}

Range Range::all_ghosts() const
{
    return with_ghosts(Ng);
}

Range Range::with_ghosts(int const NgEff) const
{
    Range rng(*this);
    if (NgEff > Ng) {
        throw std::runtime_error("NgEff > Ng");
    }
    rng.NgEff = NgEff;
    return rng;
}

std::ostream& operator<<(std::ostream& os, Range const& rng)
{
    for (int idim = 0; idim < 3; ++idim) {
        os << "[" << rng.Corner_min[idim] << "," << rng.Corner_max[idim] << "[";
        if (idim != 2) {
            os << "x";
        }
    }
    return os;
}

std::array<Kokkos::Array<int, 3>, 2> cell_range(Range const& range)
{
    Kokkos::Array<int, 3> begin;
    Kokkos::Array<int, 3> end;
    for (int idim = 0; idim < ndim; ++idim) {
        begin[idim] = range.Nghost[idim] - range.NgEff;
        end[idim] = range.Nghost[idim] + range.Corner_max[idim] - range.Corner_min[idim] + range.NgEff;
    }

    for (int idim = ndim; idim < 3; ++idim) {
        begin[idim] = range.Nghost[idim];
        end[idim] = range.Nghost[idim] + range.Corner_max[idim] - range.Corner_min[idim];
    }

    return std::array<Kokkos::Array<int, 3>, 2> {begin, end};
}

Kokkos::MDRangePolicy<Kokkos::IndexType<int>, Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>> cell_mdrange(Range const& range)
{
    auto const [begin, end] = cell_range(range);
    return {begin, end};
}

} // namespace novapp

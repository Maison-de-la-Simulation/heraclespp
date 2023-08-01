/**
 * @file range.cpp
 * Geom class implementation
 */

#include <array>
#include <iostream>
#include <stdexcept>

#include <Kokkos_Core.hpp>

#include "ndim.hpp"
#include "range.hpp"

namespace novapp
{

Range::Range(std::array<int, 3> const& Cmin, std::array<int, 3> const& Cmax, int Ng)
    : Corner_min(Cmin)
    , Corner_max(Cmax)
    , Ng(Ng)
    , NgEff(0)
    , Nghost({0, 0, 0})
{
    for (int idim = 0; idim < ndim; ++idim)
    {
        Nghost[idim] = Ng;
    }

    for (int idim = 0; idim < 3; ++idim)
    {
        if (Cmin[idim] < 0 || Cmax[idim] < 0)
        {
            throw std::runtime_error("Cmin < 0 || Cmax < 0");
        }

        if (Cmin[idim] > Cmax[idim])
        {
            throw std::runtime_error("Cmin > Cmax");
        }
    }

    for (int idim = 0; idim < ndim; ++idim)
    {
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

    for (int idim = ndim; idim < 3; ++idim)
    {
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

std::ostream& operator<<(std::ostream& os, Range const& rng)
{
    for (int idim = 0; idim < 3; ++idim)
    {
        os << "[" << rng.Corner_min[idim] << "," << rng.Corner_max[idim] << "[";
        if (idim != 2)
        {
            os << "x";
        }
    }
    return os;
}

} // namespace novapp

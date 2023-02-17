/**
 * @file range.cpp
 * Geom class implementation
 */

#include <stdexcept>

#include "range.hpp"

Range::Range(
        std::array<int, 3> const Cmin,
        std::array<int, 3> const Cmax,
        std::array<int, 3> const Nghost)
    : Corner_min(Cmin)
    , Corner_max(Cmax)
{
    for (int idim = 0; idim < 3; ++idim)
    {
        if (Cmin[idim] < 0 || Cmax[idim] < 0)
        {
            throw std::runtime_error("Cmin < 0 || Cmax < 0");
        }

        if (Nghost[idim] < 2)
        {
            throw std::runtime_error("Nghost < 2");
        }

        if (Cmin[idim] > Cmax[idim])
        {
            throw std::runtime_error("Cmin > Cmax");
        }
    }

    for (int idim = 0; idim < 3; ++idim)
    {
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
}

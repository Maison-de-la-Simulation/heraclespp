
/**
 * @file range.hpp
 * Grid class declaration
 */
#pragma once

#include <array>

class Range
{
public :

    std::array<int, 3> Corner_min ; // Bottom, left, back corner without ghost
    std::array<int, 3> Corner_max ; // top, right, front corner without ghost

    std::array<int, 3> Nc_min_2g, Nc_max_2g ; // min, max of cell centered viarables with 2 ghost cells
    std::array<int, 3> Nf_min_2g, Nf_max_2g ; // min, max of face centered viarables with 2 ghost cells

    std::array<int, 3> Nc_min_1g, Nc_max_1g ; // min, max of cell centered viarables with 1 ghost cells
    std::array<int, 3> Nf_min_1g, Nf_max_1g ; // min, max of face centered viarables with 1 ghost cells

    std::array<int, 3> Nc_min_0g, Nc_max_0g ; // min, max of cell centered viarables with 1 ghost cells
    std::array<int, 3> Nf_min_0g, Nf_max_0g ; // min, max of face centered viarables with 1 ghost cells

    void fill_range(int ndim, std::array<int, 3> const & Nghost);
    Range(){};
    Range(std::array<int, 3> Cmin, std::array<int, 3> Cmax, std::array<int, 3> Nghost);
};

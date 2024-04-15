#pragma once

#include <stdexcept>

#include <grid_type.hpp>
#include <nova_params.hpp>

namespace novapp
{

class GridSetup : public IGridType
{
public:
    explicit GridSetup([[maybe_unused]] Param const& param)
    {
    }

    void execute(
        [[maybe_unused]] std::array<int, 3> Nghost,
        [[maybe_unused]] std::array<int, 3> Nx_glob_ng,
        [[maybe_unused]] KVH_double_1d const& x_glob,
        [[maybe_unused]] KVH_double_1d const& y_glob,
        [[maybe_unused]] KVH_double_1d const& z_glob) const final
    {
        throw std::runtime_error("Grid setup not implemented");
    }
};

} // namespace novapp

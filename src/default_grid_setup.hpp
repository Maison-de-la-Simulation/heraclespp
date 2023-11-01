#pragma once

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
};

} // namespace novapp

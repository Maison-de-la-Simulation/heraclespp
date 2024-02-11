# pragma once

#include <stdexcept>

#include <eos.hpp>
#include <grid.hpp>

#include "boundary.hpp"

namespace novapp
{

class ParamSetup;

template <class Gravity>
class BoundarySetup : public IBoundaryCondition
{
public:
    BoundarySetup(int idim, int iface,
        [[maybe_unused]] EOS const& eos,
        [[maybe_unused]] Grid const& grid,
        [[maybe_unused]] ParamSetup const& param_setup,
        [[maybe_unused]] Gravity const& gravity)
        : IBoundaryCondition(idim, iface)
    {
    }

    void execute([[maybe_unused]] KV_double_3d rho,
        [[maybe_unused]] KV_double_4d rhou,
        [[maybe_unused]] KV_double_3d E,
        [[maybe_unused]] KV_double_4d fx) const final
    {
        throw std::runtime_error("Boundary setup not implemented");
    }
};

} // namespace novapp
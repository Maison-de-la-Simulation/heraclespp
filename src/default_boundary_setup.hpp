# pragma once

#include <stdexcept>

#include <eos.hpp>
#include <grid.hpp>

#include "boundary.hpp"

namespace novapp
{

class ParamSetup;

template <class Gravity>
class BoundarySetup : public IBoundaryCondition<Gravity>
{
public:
    BoundarySetup(int idim, int iface,
        [[maybe_unused]] EOS const& eos,
        [[maybe_unused]] ParamSetup const& param_setup)
        : IBoundaryCondition<Gravity>(idim, iface)
    {
    }

    void execute([[maybe_unused]] Grid const& grid,
        [[maybe_unused]] Gravity const& gravity,
        [[maybe_unused]] KV_double_3d const& rho,
        [[maybe_unused]] KV_double_4d const& rhou,
        [[maybe_unused]] KV_double_3d const& E,
        [[maybe_unused]] KV_double_4d const& fx) const final
    {
        throw std::runtime_error("Boundary setup not implemented");
    }
};

} // namespace novapp
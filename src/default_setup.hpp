# pragma once

#include <Kokkos_Core.hpp>

#include <inih/INIReader.hpp>

#include "eos.hpp"
#include <grid.hpp>
#include "initialization_interface.hpp"
#include "kokkos_shortcut.hpp"
#include "default_boundary_setup.hpp"
#include "default_grid_setup.hpp"
#include <range.hpp>

namespace novapp
{

class ParamSetup
{
public:
    explicit ParamSetup(INIReader const& reader)
    {
    }
};

template <class Gravity>
class InitializationSetup : public IInitializationProblem
{
public:
    InitializationSetup(
        EOS const& eos,
        Grid const& grid,
        ParamSetup const& param_set_up,
        Gravity const& gravity)
    {}

    void execute(
        [[maybe_unused]] Range const& range,
        [[maybe_unused]] KV_double_3d const rho,
        [[maybe_unused]] KV_double_4d const u,
        [[maybe_unused]] KV_double_3d const P,
        [[maybe_unused]] KV_double_4d const fx) const final
    {}
};

} // namespace novapp

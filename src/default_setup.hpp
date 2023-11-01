# pragma once

#include <inih/INIReader.hpp>

#include <eos.hpp>
#include <grid.hpp>
#include <kokkos_shortcut.hpp>
#include <range.hpp>

#include "default_boundary_setup.hpp"
#include "default_grid_setup.hpp"
#include "initialization_interface.hpp"

namespace novapp
{

class ParamSetup
{
public:
    explicit ParamSetup([[maybe_unused]] INIReader const& reader)
    {
    }
};

template <class Gravity>
class InitializationSetup : public IInitializationProblem
{
public:
    InitializationSetup(
        [[maybe_unused]] EOS const& eos,
        [[maybe_unused]] Grid const& grid,
        [[maybe_unused]] ParamSetup const& param_set_up,
        [[maybe_unused]] Gravity const& gravity)
    {
    }

    void execute(
        [[maybe_unused]] Range const& range,
        [[maybe_unused]] KV_double_3d const rho,
        [[maybe_unused]] KV_double_4d const u,
        [[maybe_unused]] KV_double_3d const P,
        [[maybe_unused]] KV_double_4d const fx) const final
    {
    }
};

} // namespace novapp

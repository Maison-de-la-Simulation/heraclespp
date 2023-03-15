#include <PerfectGas.hpp>

#include "array_conversion.hpp"
#include "euler_equations.hpp"
#include "range.hpp"
namespace novapp
{

void conv_prim_to_cons(
    Range const& range,
    Kokkos::View<double****, Kokkos::LayoutStride> const rhou,
    Kokkos::View<double***, Kokkos::LayoutStride> const E,
    Kokkos::View<const double***, Kokkos::LayoutStride> const rho,
    Kokkos::View<const double****, Kokkos::LayoutStride> const u,
    Kokkos::View<const double***, Kokkos::LayoutStride> const P,
    thermodynamics::PerfectGas const& eos)
{
    auto const [begin, end] = cell_range(range);
    Kokkos::parallel_for(
    "ConvPrimtoConsArray",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
    KOKKOS_LAMBDA(int i, int j, int k)
    {
        EulerPrim var_prim;
        var_prim.density = rho(i, j, k);
        var_prim.pressure = P(i, j, k);
        for (int idim = 0; idim < ndim; ++idim)
        {
            var_prim.velocity[idim] = u(i, j, k, idim);
        }
        EulerCons cons = to_cons(var_prim, eos);
        E(i, j, k) = cons.energy;
        for (int idim = 0; idim < ndim; ++idim)
        {
            rhou(i, j, k, idim) = cons.momentum[idim];
        }
    });
}
 
void conv_cons_to_prim(
    Range const& range,
    KV_double_4d const u,
    KV_double_3d const P,
    KV_cdouble_3d const rho,
    KV_cdouble_4d const rhou,
    KV_cdouble_3d const E,
    thermodynamics::PerfectGas const& eos)
{
    auto const [begin, end] = cell_range(range);
     Kokkos::parallel_for(
    "ConvConstoPrimArray",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
    KOKKOS_LAMBDA(int i, int j, int k)
    {
        EulerCons var_cons;
        var_cons.density = rho(i, j, k);
        var_cons.energy = E(i, j, k);
        for (int idim = 0; idim < ndim; ++idim)
        {
            var_cons.momentum[idim] = rhou(i, j, k, idim);
        }
        EulerPrim prim = to_prim(var_cons, eos);
        P(i, j, k) = prim.pressure;
        for (int idim = 0; idim < ndim; ++idim)
        {
            u(i, j, k, idim) = prim.velocity[idim];
        }
    });   
}

} // namespace novapp

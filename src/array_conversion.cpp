#include "array_conversion.hpp"

namespace novapp
{
    
void conv_prim_to_cons(
    Range const& range,
    Kokkos::View<double****, Kokkos::LayoutStride> const rhou,
    Kokkos::View<double***, Kokkos::LayoutStride> const E,
    Kokkos::View<const double***, Kokkos::LayoutStride> const rho,
    Kokkos::View<const double****, Kokkos::LayoutStride> const u,
    Kokkos::View<const double***, Kokkos::LayoutStride> const P,
    EOS const& eos)
{
    auto const [begin, end] = cell_range(range);
    Kokkos::parallel_for(
    "ConvPrimtoConsArray",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
    KOKKOS_LAMBDA(int i, int j, int k)
    {
        EulerPrim var_prim;
        var_prim.rho = rho(i, j, k);
        var_prim.P = P(i, j, k);
        for (int idim = 0; idim < ndim; ++idim)
        {
            var_prim.u[idim] = u(i, j, k, idim);
        }
        EulerCons cons = to_cons(var_prim, eos);
        E(i, j, k) = cons.E;
        for (int idim = 0; idim < ndim; ++idim)
        {
            rhou(i, j, k, idim) = cons.rhou[idim];
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
    EOS const& eos)
{
    auto const [begin, end] = cell_range(range);
     Kokkos::parallel_for(
    "ConvConstoPrimArray",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
    KOKKOS_LAMBDA(int i, int j, int k)
    {
        EulerCons var_cons;
        var_cons.rho = rho(i, j, k);
        var_cons.E = E(i, j, k);
        for (int idim = 0; idim < ndim; ++idim)
        {
            var_cons.rhou[idim] = rhou(i, j, k, idim);
        }
        EulerPrim prim = to_prim(var_cons, eos);
        P(i, j, k) = prim.P;
        for (int idim = 0; idim < ndim; ++idim)
        {
            u(i, j, k, idim) = prim.u[idim];
        }
    });
}

} // namespace novapp

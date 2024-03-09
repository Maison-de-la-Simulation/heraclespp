#include <Kokkos_Core.hpp>
#include <grid.hpp>
#include <kokkos_shortcut.hpp>
#include <range.hpp>

#include "broadcast.hpp"

namespace novapp {

void broadcast(
        Range const& range,
        [[maybe_unused]] Grid const& grid,
        double const in,
        KV_double_3d const out)
{
    Kokkos::parallel_for(
            "broadcast 0d->3d",
            cell_mdrange(range),
            KOKKOS_LAMBDA(const int i, const int j, const int k) { out(i, j, k) = in; });
}

void broadcast(Range const& range, Grid const& grid, KV_cdouble_1d const in, KV_double_3d const out)
{
    assert(in.extent_int(0) + 2 * grid.Ng == out.extent_int(0));
    int const ghost_x = grid.Nghost[0];

    Kokkos::parallel_for(
            "broadcast 1d->3d",
            cell_mdrange(range),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                out(i, j, k) = in(i - ghost_x);
            });
}

} // namespace novapp

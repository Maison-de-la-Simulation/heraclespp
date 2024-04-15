#include <Kokkos_Core.hpp>
#include <kokkos_shortcut.hpp>
#include <range.hpp>

#include "broadcast.hpp"

namespace novapp {

void broadcast(Range const& range, double const in, KV_double_3d const& out)
{
    assert(range.NgEff == 0);
    Kokkos::parallel_for(
            "broadcast 0d->3d",
            cell_mdrange(range),
            KOKKOS_LAMBDA(const int i, const int j, const int k) { out(i, j, k) = in; });
}

void broadcast(Range const& range, KV_cdouble_1d const& in, KV_double_3d const& out)
{
    assert(range.NgEff == 0);
    assert(in.extent_int(0) + 2 * range.Nghost[0] == out.extent_int(0));
    int const ghost_x = range.Nghost[0];

    Kokkos::parallel_for(
            "broadcast 1d->3d",
            cell_mdrange(range),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                out(i, j, k) = in(i - ghost_x);
            });
}

} // namespace novapp

#include <gtest/gtest.h>

#include <ndim.hpp>
#include <mesh/range.hpp>

constexpr std::array<int, 3> dim_mask(
        std::array<int, 3> const lhs,
        std::array<int, 3> const rhs) noexcept
{
    std::array<int, 3> out = lhs;
    for (int idim = novapp::ndim; idim < 3; ++idim)
    {
        out[idim] = rhs[idim];
    }
    return out;
}

TEST(Range, Constructor)
{
    // EXPECT_THROW(Range({0, 0, 0}, {0, 0, 0}, dim_mask({0, 2, 2}, {0, 0, 0})), std::runtime_error);
    // if (novapp::ndim >= 2)
    // {
    //     EXPECT_THROW(
    //             Range({0, 0, 0}, {0, 0, 0}, dim_mask({2, 0, 2}, {0, 0, 0})),
    //             std::runtime_error);
    // }
    // if (novapp::ndim == 3)
    // {
    //     EXPECT_THROW(
    //             Range({0, 0, 0}, {0, 0, 0}, dim_mask({2, 2, 0}, {0, 0, 0})),
    //             std::runtime_error);
    // }

    EXPECT_THROW(novapp::Range({1, 0, 0}, {0, 0, 0}, 2), std::runtime_error);
    EXPECT_THROW(novapp::Range({0, 1, 0}, {0, 0, 0}, 2), std::runtime_error);
    EXPECT_THROW(novapp::Range({0, 0, 1}, {0, 0, 0}, 2), std::runtime_error);

    EXPECT_THROW(novapp::Range({-1, 0, 0}, {0, 0, 0}, 2), std::runtime_error);
    EXPECT_THROW(novapp::Range({0, -1, 0}, {0, 0, 0}, 2), std::runtime_error);
    EXPECT_THROW(novapp::Range({0, 0, -1}, {0, 0, 0}, 2), std::runtime_error);

    EXPECT_THROW(novapp::Range({0, 0, 0}, {-1, 0, 0}, 2), std::runtime_error);
    EXPECT_THROW(novapp::Range({0, 0, 0}, {0, -1, 0}, 2), std::runtime_error);
    EXPECT_THROW(novapp::Range({0, 0, 0}, {0, 0, -1}, 2), std::runtime_error);
}

TEST(Range, Accessors)
{
    std::array<int, 3> const Cmin {1, 1, 1};
    std::array<int, 3> const Cmax {2, 2, 2};
    int const Ng = 2;
    novapp::Range const rng(Cmin, Cmax, Ng);
    EXPECT_EQ(rng.Corner_min, Cmin);
    EXPECT_EQ(rng.Corner_max, Cmax);
}

TEST(Range, Empty)
{
    std::array<int, 3> const Cmin {2, 3, 4};
    std::array<int, 3> const Cmax = Cmin;
    int const Ng = 3;
    novapp::Range const rng(Cmin, Cmax, Ng);
    EXPECT_EQ(rng.Nc_min_0g, dim_mask({3, 3, 3}, {0, 0, 0}));
    EXPECT_EQ(rng.Nc_max_0g, dim_mask({3, 3, 3}, {0, 0, 0}));
}

TEST(Range, CellBounds)
{
    std::array<int, 3> const Cmin {2, 3, 4};
    std::array<int, 3> const Cmax {4, 5, 6};
    int const Ng = 3;
    novapp::Range const rng(Cmin, Cmax, Ng);
    EXPECT_EQ(rng.Nc_min_0g, dim_mask({3, 3, 3}, {0, 0, 0}));
    EXPECT_EQ(rng.Nc_max_0g, dim_mask({5, 5, 5}, {2, 2, 2}));
    EXPECT_EQ(rng.Nc_min_1g, dim_mask({2, 2, 2}, {0, 0, 0}));
    EXPECT_EQ(rng.Nc_max_1g, dim_mask({6, 6, 6}, {2, 2, 2}));
    EXPECT_EQ(rng.Nc_min_2g, dim_mask({1, 1, 1}, {0, 0, 0}));
    EXPECT_EQ(rng.Nc_max_2g, dim_mask({7, 7, 7}, {2, 2, 2}));
}

TEST(Range, FaceBounds)
{
    std::array<int, 3> const Cmin {2, 3, 4};
    std::array<int, 3> const Cmax {4, 5, 6};
    int const Ng = 3;
    novapp::Range const rng(Cmin, Cmax, Ng);
    EXPECT_EQ(rng.Nf_min_0g, dim_mask({3, 3, 3}, {0, 0, 0}));
    EXPECT_EQ(rng.Nf_max_0g, dim_mask({6, 6, 6}, {3, 3, 3}));
    EXPECT_EQ(rng.Nf_min_1g, dim_mask({2, 2, 2}, {0, 0, 0}));
    EXPECT_EQ(rng.Nf_max_1g, dim_mask({7, 7, 7}, {3, 3, 3}));
    EXPECT_EQ(rng.Nf_min_2g, dim_mask({1, 1, 1}, {0, 0, 0}));
    EXPECT_EQ(rng.Nf_max_2g, dim_mask({8, 8, 8}, {3, 3, 3}));
}

#include <gtest/gtest.h>

#include <range.hpp>

TEST(Range, Constructor)
{
    EXPECT_THROW(Range({0, 0, 0}, {0, 0, 0}, {1, 2, 2}), std::runtime_error);
    EXPECT_THROW(Range({0, 0, 0}, {0, 0, 0}, {2, 1, 2}), std::runtime_error);
    EXPECT_THROW(Range({0, 0, 0}, {0, 0, 0}, {2, 2, 1}), std::runtime_error);

    EXPECT_THROW(Range({1, 0, 0}, {0, 0, 0}, {2, 2, 2}), std::runtime_error);
    EXPECT_THROW(Range({0, 1, 0}, {0, 0, 0}, {2, 2, 2}), std::runtime_error);
    EXPECT_THROW(Range({0, 0, 1}, {0, 0, 0}, {2, 2, 2}), std::runtime_error);

    EXPECT_THROW(Range({-1, 0, 0}, {0, 0, 0}, {2, 2, 2}), std::runtime_error);
    EXPECT_THROW(Range({0, -1, 0}, {0, 0, 0}, {2, 2, 2}), std::runtime_error);
    EXPECT_THROW(Range({0, 0, -1}, {0, 0, 0}, {2, 2, 2}), std::runtime_error);

    EXPECT_THROW(Range({0, 0, 0}, {-1, 0, 0}, {2, 2, 2}), std::runtime_error);
    EXPECT_THROW(Range({0, 0, 0}, {0, -1, 0}, {2, 2, 2}), std::runtime_error);
    EXPECT_THROW(Range({0, 0, 0}, {0, 0, -1}, {2, 2, 2}), std::runtime_error);
}

TEST(Range, Accessors)
{
    std::array<int, 3> const Cmin {1, 1, 1};
    std::array<int, 3> const Cmax {2, 2, 2};
    std::array<int, 3> const Nghost {2, 2, 2};
    Range const rng(Cmin, Cmax, Nghost);
    EXPECT_EQ(rng.Corner_min, Cmin);
    EXPECT_EQ(rng.Corner_max, Cmax);
}

TEST(Range, Empty)
{
    std::array<int, 3> const Cmin {2, 3, 4};
    std::array<int, 3> const Cmax = Cmin;
    std::array<int, 3> const Nghost {2, 3, 3};
    Range const rng(Cmin, Cmax, Nghost);
    EXPECT_EQ(rng.Nc_min_0g, (std::array<int, 3> {2, 3, 3}));
    EXPECT_EQ(rng.Nc_max_0g, (std::array<int, 3> {2, 3, 3}));
}

TEST(Range, CellBounds)
{
    std::array<int, 3> const Cmin {2, 3, 4};
    std::array<int, 3> const Cmax {4, 5, 6};
    std::array<int, 3> const Nghost {2, 3, 3};
    Range const rng(Cmin, Cmax, Nghost);
    EXPECT_EQ(rng.Nc_min_0g, (std::array<int, 3> {2, 3, 3}));
    EXPECT_EQ(rng.Nc_max_0g, (std::array<int, 3> {4, 5, 5}));
    EXPECT_EQ(rng.Nc_min_1g, (std::array<int, 3> {1, 2, 2}));
    EXPECT_EQ(rng.Nc_max_1g, (std::array<int, 3> {5, 6, 6}));
    EXPECT_EQ(rng.Nc_min_2g, (std::array<int, 3> {0, 1, 1}));
    EXPECT_EQ(rng.Nc_max_2g, (std::array<int, 3> {6, 7, 7}));
}

TEST(Range, FaceBounds)
{
    std::array<int, 3> const Cmin {2, 3, 4};
    std::array<int, 3> const Cmax {4, 5, 6};
    std::array<int, 3> const Nghost {2, 3, 3};
    Range const rng(Cmin, Cmax, Nghost);
    EXPECT_EQ(rng.Nf_min_0g, (std::array<int, 3> {2, 3, 3}));
    EXPECT_EQ(rng.Nf_max_0g, (std::array<int, 3> {5, 6, 6}));
    EXPECT_EQ(rng.Nf_min_1g, (std::array<int, 3> {1, 2, 2}));
    EXPECT_EQ(rng.Nf_max_1g, (std::array<int, 3> {6, 7, 7}));
    EXPECT_EQ(rng.Nf_min_2g, (std::array<int, 3> {0, 1, 1}));
    EXPECT_EQ(rng.Nf_max_2g, (std::array<int, 3> {7, 8, 8}));
}

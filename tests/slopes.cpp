#include <gtest/gtest.h>

#include <slope_limiters.hpp>

TEST(NullSlopeLimiter, Zero)
{
    double const diffR = 68.9;
    double const diffL = 37.65;
    novapp::Constant const slope;
    EXPECT_EQ(slope(diffR, diffL), 0.);
}

template <class SlopeLimiter>
class SlopeLimiterFixture : public ::testing::Test
{
public:
    SlopeLimiter m_slope_limiter;

    SlopeLimiterFixture() = default;

    SlopeLimiterFixture(SlopeLimiterFixture const& rhs) = default;

    SlopeLimiterFixture(SlopeLimiterFixture&& rhs) noexcept = default;

    ~SlopeLimiterFixture() override = default;

    SlopeLimiterFixture& operator=(SlopeLimiterFixture const& rhs) = default;

    SlopeLimiterFixture& operator=(SlopeLimiterFixture&& rhs) noexcept = default;
};

using SlopeLimiters = ::testing::Types<novapp::Minmod, novapp::VanAlbada, novapp::VanLeer>;
// Trailing comma is needed to avoid spurious `gnu-zero-variadic-macro-arguments` warning with clang
TYPED_TEST_SUITE(SlopeLimiterFixture, SlopeLimiters, );

TYPED_TEST(SlopeLimiterFixture, NullOnOppositeSign)
{
    double const diffR = 68.9;
    double const diffL = 37.65;
    EXPECT_DOUBLE_EQ(this->m_slope_limiter(+diffR, -diffL), 0.);
    EXPECT_DOUBLE_EQ(this->m_slope_limiter(-diffR, +diffL), 0.);
}

TYPED_TEST(SlopeLimiterFixture, Identity)
{
    double const diff = 68.9;
    EXPECT_DOUBLE_EQ(this->m_slope_limiter(+diff, +diff), +diff);
    EXPECT_DOUBLE_EQ(this->m_slope_limiter(-diff, -diff), -diff);
}

TYPED_TEST(SlopeLimiterFixture, Symmetry)
{
    double const diffR = 2.;
    double const diffL = 1.;
    EXPECT_DOUBLE_EQ(this->m_slope_limiter(+diffR, +diffL), this->m_slope_limiter(+diffL, +diffR));
    EXPECT_DOUBLE_EQ(this->m_slope_limiter(-diffR, -diffL), this->m_slope_limiter(-diffL, -diffR));
}

// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include <perfect_gas.hpp>
#include <rad_gas.hpp>

template <class EoS>
class EquationOfStateFixture : public ::testing::Test
{
public:
    using eos_type = EoS;

    EquationOfStateFixture() = default;

    EquationOfStateFixture(EquationOfStateFixture const& rhs) = default;

    EquationOfStateFixture(EquationOfStateFixture&& rhs) noexcept = default;

    ~EquationOfStateFixture() override = default;

    auto operator=(EquationOfStateFixture const& rhs) -> EquationOfStateFixture& = default;

    auto operator=(EquationOfStateFixture&& rhs) noexcept -> EquationOfStateFixture& = default;
};

using EquationOfStates = ::testing::Types<hclpp::thermodynamics::PerfectGas, hclpp::thermodynamics::RadGas>;
// Trailing comma is needed to avoid spurious `gnu-zero-variadic-macro-arguments` warning with clang
TYPED_TEST_SUITE(EquationOfStateFixture, EquationOfStates, );

TYPED_TEST(EquationOfStateFixture, GammaValidityRange)
{
    using eos_t = typename TestFixture::eos_type;
    std::vector<double> const valid_values {std::nextafter(1., std::numeric_limits<double>::infinity()), std::numeric_limits<double>::max()};

    std::vector<double> invalid_values {-std::numeric_limits<double>::infinity(), -1., +1., +std::numeric_limits<double>::infinity()};

    if (std::numeric_limits<double>::has_quiet_NaN) {
        invalid_values.emplace_back(std::numeric_limits<double>::quiet_NaN());
    }

    if (std::numeric_limits<double>::has_signaling_NaN) {
        invalid_values.emplace_back(std::numeric_limits<double>::signaling_NaN());
    }

    double const valid_mmw = 1;
    for (double const invalid_gamma : invalid_values) {
        EXPECT_THROW(eos_t(invalid_gamma, valid_mmw), std::domain_error);
    }
    for (double const valid_gamma : valid_values) {
        EXPECT_NO_THROW(eos_t(valid_gamma, valid_mmw));
    }
}

TYPED_TEST(EquationOfStateFixture, MmwValidityRange)
{
    using eos_t = typename TestFixture::eos_type;
    std::vector<double> const valid_values {std::numeric_limits<double>::denorm_min(), +1., std::numeric_limits<double>::max()};

    std::vector<double> invalid_values {-std::numeric_limits<double>::infinity(), -1., 0., +std::numeric_limits<double>::infinity()};

    if (std::numeric_limits<double>::has_quiet_NaN) {
        invalid_values.emplace_back(std::numeric_limits<double>::quiet_NaN());
    }

    if (std::numeric_limits<double>::has_signaling_NaN) {
        invalid_values.emplace_back(std::numeric_limits<double>::signaling_NaN());
    }

    double const valid_gamma = 1.4;
    for (double const invalid_mmw : invalid_values) {
        EXPECT_THROW(eos_t(valid_gamma, invalid_mmw), std::domain_error);
    }
    for (double const valid_mmw : valid_values) {
        EXPECT_NO_THROW(eos_t(valid_gamma, valid_mmw));
    }
}

TYPED_TEST(EquationOfStateFixture, Accessors)
{
    using eos_t = typename TestFixture::eos_type;
    double const gamma = 1.4;
    double const mmw = 1;

    eos_t const eos(gamma, mmw);
    EXPECT_DOUBLE_EQ(eos.adiabatic_index(), gamma);
    EXPECT_DOUBLE_EQ(eos.mean_molecular_weight(), mmw);
}

TYPED_TEST(EquationOfStateFixture, ValidState)
{
    using eos_t = typename TestFixture::eos_type;
    double const gamma = 1.4;
    double const mmw = 1;
    eos_t const eos(gamma, mmw);

    std::vector<double> const
            valid_values {std::numeric_limits<double>::denorm_min(), std::numeric_limits<double>::min(), 1., std::numeric_limits<double>::max()};

    std::vector<double> invalid_values {-std::numeric_limits<double>::infinity(), -1., 0., +std::numeric_limits<double>::infinity()};

    if (std::numeric_limits<double>::has_quiet_NaN) {
        invalid_values.emplace_back(std::numeric_limits<double>::quiet_NaN());
    }

    if (std::numeric_limits<double>::has_signaling_NaN) {
        invalid_values.emplace_back(std::numeric_limits<double>::signaling_NaN());
    }

    for (double const valid_density : valid_values) {
        for (double const valid_pressure : valid_values) {
            EXPECT_TRUE(eos.is_valid(valid_density, valid_pressure));
        }
    }

    for (double const invalid_v1 : invalid_values) {
        for (double const invalid_v2 : invalid_values) {
            EXPECT_FALSE(eos.is_valid(invalid_v1, invalid_v2));
            EXPECT_FALSE(eos.is_valid(invalid_v2, invalid_v1));
        }
    }
}

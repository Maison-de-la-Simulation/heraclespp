#include <stdexcept>

#include <gtest/gtest.h>

#include <PerfectGas.hpp>

TEST(Thermodynamics, GammaValidityRange)
{
    std::vector<double> const valid_values {
            std::nextafter(1., std::numeric_limits<double>::infinity()),
            std::numeric_limits<double>::max()};

    std::vector<double> invalid_values {
            -std::numeric_limits<double>::infinity(),
            -1.,
            +1.,
            +std::numeric_limits<double>::infinity()};

    if (std::numeric_limits<double>::has_quiet_NaN)
    {
        invalid_values.emplace_back(std::numeric_limits<double>::quiet_NaN());
    }

    if (std::numeric_limits<double>::has_signaling_NaN)
    {
        invalid_values.emplace_back(std::numeric_limits<double>::signaling_NaN());
    }

    double const valid_mmw = 1;
    for (double const invalid_gamma : invalid_values)
    {
        EXPECT_THROW(novapp::thermodynamics::PerfectGas(invalid_gamma, valid_mmw), std::domain_error);
    }
    for (double const valid_gamma : valid_values)
    {
        EXPECT_NO_THROW(novapp::thermodynamics::PerfectGas(valid_gamma, valid_mmw));
    }
}

TEST(Thermodynamics, MmwValidityRange)
{
    std::vector<double> const valid_values {
            std::numeric_limits<double>::denorm_min(),
            +1.,
            std::numeric_limits<double>::max()};

    std::vector<double> invalid_values {
            -std::numeric_limits<double>::infinity(),
            -1.,
            0.,
            +std::numeric_limits<double>::infinity()};

    if (std::numeric_limits<double>::has_quiet_NaN)
    {
        invalid_values.emplace_back(std::numeric_limits<double>::quiet_NaN());
    }

    if (std::numeric_limits<double>::has_signaling_NaN)
    {
        invalid_values.emplace_back(std::numeric_limits<double>::signaling_NaN());
    }

    double const valid_gamma = 1.4;
    for (double const invalid_mmw : invalid_values)
    {
        EXPECT_THROW(novapp::thermodynamics::PerfectGas(valid_gamma, invalid_mmw), std::domain_error);
    }
    for (double const valid_mmw : valid_values)
    {
        EXPECT_NO_THROW(novapp::thermodynamics::PerfectGas(valid_gamma, valid_mmw));
    }
}

TEST(Thermodynamics, Accessors)
{
    double const gamma = 1.4;
    double const mmw = 1;
    novapp::thermodynamics::PerfectGas const eos(gamma, mmw);
    EXPECT_DOUBLE_EQ(eos.adiabatic_index(), gamma);
    EXPECT_DOUBLE_EQ(eos.mean_molecular_weight(), mmw);
}

TEST(Thermodynamics, ValidState)
{
    double const gamma = 1.4;
    double const mmw = 1;
    novapp::thermodynamics::PerfectGas const eos(gamma, mmw);

    std::vector<double> const valid_values {
            std::numeric_limits<double>::denorm_min(),
            std::numeric_limits<double>::min(),
            1.,
            std::numeric_limits<double>::max()};

    std::vector<double> invalid_values {
            -std::numeric_limits<double>::infinity(),
            -1.,
            0.,
            +std::numeric_limits<double>::infinity()};

    if (std::numeric_limits<double>::has_quiet_NaN)
    {
        invalid_values.emplace_back(std::numeric_limits<double>::quiet_NaN());
    }

    if (std::numeric_limits<double>::has_signaling_NaN)
    {
        invalid_values.emplace_back(std::numeric_limits<double>::signaling_NaN());
    }

    for (double const valid_density : valid_values)
    {
        for (double const valid_pressure : valid_values)
        {
            EXPECT_TRUE(eos.is_valid(valid_density, valid_pressure));
        }
    }

    for (double const invalid_v1 : invalid_values)
    {
        for (double const invalid_v2 : invalid_values)
        {
            EXPECT_FALSE(eos.is_valid(invalid_v1, invalid_v2));
            EXPECT_FALSE(eos.is_valid(invalid_v2, invalid_v1));
        }
    }
}

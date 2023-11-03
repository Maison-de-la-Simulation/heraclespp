//!
//! @file user_step.hpp
//!

#pragma once

#include <kokkos_shortcut.hpp>

namespace novapp
{

class Range;

class IUserStep
{
public:
    IUserStep();

    IUserStep(IUserStep const& rhs);

    IUserStep(IUserStep&& rhs) noexcept;

    virtual ~IUserStep() noexcept;

    IUserStep& operator=(IUserStep const& rhs);

    IUserStep& operator=(IUserStep&& rhs) noexcept;

    virtual void execute(
        Range const &range,
        double t,
        double dt,
        KV_double_3d rho,
        KV_double_3d E,
        KV_double_4d fx) const;
};

class NoUserStep : public IUserStep
{
public:
    void execute(
        Range const &range,
        double t,
        double dt,
        KV_double_3d rho,
        KV_double_3d E,
        KV_double_4d fx) const final;
};

class HeatNickelStep : public IUserStep
{
public:
    void execute(
        Range const &range,
        double t,
        double dt,
        KV_double_3d rho,
        KV_double_3d E,
        KV_double_4d fx) const final;
};

} // namespace novapp

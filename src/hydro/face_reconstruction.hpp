// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file face_reconstruction.hpp
//!

#pragma once

#include <memory>
#include <string>

#include <kokkos_shortcut.hpp>

namespace hclpp {

class Grid;
class Range;

class IFaceReconstruction
{
public:
    IFaceReconstruction();

    IFaceReconstruction(IFaceReconstruction const& rhs);

    IFaceReconstruction(IFaceReconstruction&& rhs) noexcept;

    virtual ~IFaceReconstruction() noexcept;

    auto operator=(IFaceReconstruction const& rhs) -> IFaceReconstruction&;

    auto operator=(IFaceReconstruction&& rhs) noexcept -> IFaceReconstruction&;

    //! @param[in] range output iteration range
    //! @param[in] grid provides grid information
    //! @param[in] var cell values
    //! @param[out] var_rec reconstructed values at interfaces
    virtual void execute(Range const& range, Grid const& grid, KV_cdouble_3d const& var, KV_double_5d const& var_rec) const = 0;
};

auto factory_face_reconstruction(std::string const& slope) -> std::unique_ptr<IFaceReconstruction>;

} // namespace hclpp

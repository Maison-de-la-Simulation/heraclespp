// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file face_reconstruction.cpp
//!

#include <memory>
#include <stdexcept>
#include <string>

#include "face_reconstruction.hpp"
#include "limited_linear_reconstruction.hpp"
#include "slope_limiters.hpp"

namespace hclpp {

IFaceReconstruction::IFaceReconstruction() = default;

IFaceReconstruction::IFaceReconstruction(IFaceReconstruction const& rhs) = default;

IFaceReconstruction::IFaceReconstruction(IFaceReconstruction&& rhs) noexcept = default;

IFaceReconstruction::~IFaceReconstruction() noexcept = default;

auto IFaceReconstruction::operator=(IFaceReconstruction const& /*rhs*/) -> IFaceReconstruction& = default;

auto IFaceReconstruction::operator=(IFaceReconstruction&& /*rhs*/) noexcept -> IFaceReconstruction& = default;

auto factory_face_reconstruction(std::string const& slope) -> std::unique_ptr<IFaceReconstruction>
{
    if (slope == "Constant") {
        return std::make_unique<LimitedLinearReconstruction<Constant>>(Constant());
    }

    if (slope == "VanLeer") {
        return std::make_unique<LimitedLinearReconstruction<VanLeer>>(VanLeer());
    }

    if (slope == "Minmod") {
        return std::make_unique<LimitedLinearReconstruction<Minmod>>(Minmod());
    }

    if (slope == "VanAlbada") {
        return std::make_unique<LimitedLinearReconstruction<VanAlbada>>(VanAlbada());
    }

    throw std::runtime_error("Unknown face reconstruction algorithm: " + slope + ".");
}

} // namespace hclpp

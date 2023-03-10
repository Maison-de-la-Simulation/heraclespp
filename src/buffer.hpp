/**
 * @file buffer.hpp
 * buffer class declaration
 */

#pragma once

#include <cmath>
#include <array>
#include <string>
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

namespace novapp
{

class Buffer
{
public :
    std::array<int, 3> Nghost;
    std::array<std::array<Kokkos::DualView<double ****>,2>,3> faceBuffer;
    std::array<std::array<Kokkos::DualView<double ****>,4>,3> edgeBuffer;
    std::array<std::array<Kokkos::DualView<double ****>,4>,2> cornerBuffer;

public :
    Buffer() = default;

    //!
    //! @fn Constructs buffer used for border exchange
    //! @brief Allocates buffer of type Kokkos::DualView.
    //! @param[in] nvar Number of variables needed for border exchanged.
    //!
    Buffer(std::array<int, 3> const & ng,
           std::array<int, 3> const & nx_local_ng,
           int nvar);
};

} // namespace novapp

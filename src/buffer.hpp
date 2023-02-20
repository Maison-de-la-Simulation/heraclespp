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

class Buffer
{
public :
    std::array<int, 3> Nghost;
    std::array<std::array<Kokkos::DualView<double ****>,2>,3> faceBuffer;
    std::array<std::array<Kokkos::DualView<double ****>,4>,3> edgeBuffer;
    std::array<std::array<Kokkos::DualView<double ****>,4>,2> cornerBuffer;

    std::array<int, 3> faceBufferSize;
    std::array<int, 3> edgeBufferSize;
    int                cornerBufferSize;
public :
    //!
    //! @fn Constructs buffer used for border exchange
    //! @brief Allocates buffer of type Kokkos::DualView.
    //! @param[in] nvar Number of variables needed for border exchanged.
    //!
    Buffer(int *ng, int *nx_local_ng, int nvar);
};

//!
//! @file buffer.hpp
//! buffer class declaration
//!

#pragma once

#include <array>
#include <string>

#include "Kokkos_shortcut.hpp"

namespace novapp
{

class Buffer
{
public :
    std::array<int, 3> Nghost;
    std::array<std::array<KDV_double_4d,2>,3> faceBuffer;
    std::array<std::array<KDV_double_4d,4>,3> edgeBuffer;
    std::array<std::array<KDV_double_4d,4>,2> cornerBuffer;

    KDV_double_4d BC_buffer;

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

    Buffer(std::array<int, 3> const & ng,
           std::array<int, 3> const & nx_local_wg,
           int idim, 
           int nvar);
};

} // namespace novapp

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
#include "grid.hpp"

class Buffer
{
public :
    int Nghost;
    std::array<std::array<Kokkos::DualView<double ****>,2>,3> faceBuffer;
    std::array<std::array<std::array<Kokkos::DualView<double ****>,2>,2>,3> edgeBuffer;
    std::array<std::array<std::array<Kokkos::DualView<double ****>,2>,2>,2> cornerBuffer;
    
    std::array<int, 3> faceBufferSize;
    std::array<int, 3> edgeBufferSize;
    int                cornerBufferSize;
public :
    //!
    //! @fn Constructs buffer used for border exchange
    //! @brief Allocates buffer pf type Kokkos::DualView.
    //! @param[in] grid Grid of simulation.
    //! @param[in] nvar Number of variables needed for border exchanged.
    //!
    Buffer(Grid *grid, int nvar);
};

//! Update buffer from device data before border exchange
//! @param[in] view Variable field device data
//! @param[in] send_buffer Buffer used to store and later send the border value
//! @param[in] ivar Index of input variable
void copyToBuffer(Kokkos::View<double***> view, Buffer *send_buffer, int ivar);


//! Update face buffer from device data before border exchange
//! @param[in] view Variable field device data
//! @param[in] send_buffer Buffer used to store and later send the border value
//! @param[in] ivar Index of input variable
void copyToBuffer_faces(Kokkos::View<double***> view, Buffer *send_buffer, int ivar);


//! Update edge buffer from device data before border exchange
//! @param[in] view Variable field device data
//! @param[in] send_buffer Buffer used to store and later send the border value
//! @param[in] ivar Index of input variable
void copyToBuffer_edges(Kokkos::View<double***> view, Buffer *send_buffer, int ivar);


//! Update corner buffer from device data before border exchange
//! @param[in] view Variable field device data
//! @param[in] send_buffer Buffer used to store and later send the border value
//! @param[in] ivar Index of input variable
void copyToBuffer_corners(Kokkos::View<double***> view, Buffer *send_buffer, int ivar);


//! Update device data from buffer after border exchange
//! @param[in] view Variable field device data
//! @param[in] recv_buffer Buffer used to receive border value
//! @param[in] ivar index of variable field
void copyFromBuffer(Kokkos::View<double***> view, Buffer *recv_buffer, int ivar);


//! Update device data from face buffer after border exchange
//! @param[in] view Variable field device data
//! @param[in] recv_buffer Buffer used to receive border value
//! @param[in] ivar index of variable field
void copyFromBuffer_faces(Kokkos::View<double***> view, Buffer *recv_buffer, int ivar);


//! Update device data from edge buffer after border exchange
//! @param[in] view Variable field device data
//! @param[in] recv_buffer Buffer used to receive border value
//! @param[in] ivar index of variable field
void copyFromBuffer_edges(Kokkos::View<double***> view, Buffer *recv_buffer, int ivar);


//! Update device data from corner buffer after border exchange
//! @param[in] view Variable field device data
//! @param[in] recv_buffer Buffer used to receive border value
//! @param[in] ivar index of variable field
void copyFromBuffer_corners(Kokkos::View<double***> view, Buffer *recv_buffer, int ivar);


//! Performe the border value exchange
//! @param[in] send_buffer Buffer used to send border value
//! @param[in] recv_buffer Buffer used to receive border value
//! @param[in] grid Grid of simulation.
void exchangeBuffer(Buffer *send_buffer, Buffer *recv_buffer, Grid *grid);


//! Performe the border face value exchange
//! @param[in] send_buffer Buffer used to send border value
//! @param[in] recv_buffer Buffer used to receive border value
//! @param[in] grid Grid of simulation.
void exchangeBuffer_faces(Buffer *send_buffer, Buffer *recv_buffer, Grid *grid);


//! Performe the border ege value exchange
//! @param[in] send_buffer Buffer used to send border value
//! @param[in] recv_buffer Buffer used to receive border value
//! @param[in] grid Grid of simulation.
void exchangeBuffer_edges(Buffer *send_buffer, Buffer *recv_buffer, Grid *grid);


//! Performe the border corner value exchange
//! @param[in] send_buffer Buffer used to send border value
//! @param[in] recv_buffer Buffer used to receive border value
//! @param[in] grid Grid of simulation.
void exchangeBuffer_corners(Buffer *send_buffer, Buffer *recv_buffer, Grid *grid);

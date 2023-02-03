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
      
enum CORNERS
{
      LOWER_LEFT_FRONT  = 0,
      LOWER_RIGHT_FRONT = 1,
      UPPER_LEFT_FRONT  = 2,
      UPPER_RIGHT_FRONT = 3,
      LOWER_LEFT_BACK   = 4,
      LOWER_RIGHT_BACK  = 5,
      UPPER_LEFT_BACK   = 6,
      UPPER_RIGHT_BACK  = 7,
};


enum EDGES
{
      EDGE01 = 0,
      EDGE23 = 1,
      EDGE45 = 2,
      EDGE67 = 3,
      EDGE02 = 4,
      EDGE13 = 5,
      EDGE46 = 6,
      EDGE57 = 7,
      EDGE04 = 8,
      EDGE15 = 9,
      EDGE26 = 10,
      EDGE37 = 11,
};

enum FACES
{
      LEFT=0,
      RIGHT=1,
      LOWER=2,
      UPPER=3,
      FRONT=4,
      BACK=5,
};


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
      Buffer(Grid *grid, int nvar);
};


void copyToBuffer        (Kokkos::View<double***> view, Buffer *send_buffer, int ivar);
void copyToBuffer_faces  (Kokkos::View<double***> view, Buffer *send_buffer, int ivar);
void copyToBuffer_edges  (Kokkos::View<double***> view, Buffer *send_buffer, int ivar);
void copyToBuffer_corners(Kokkos::View<double***> view, Buffer *send_buffer, int ivar);

void copyFromBuffer        (Kokkos::View<double***> view, Buffer *recv_buffer, int ivar);
void copyFromBuffer_faces  (Kokkos::View<double***> view, Buffer *recv_buffer, int ivar);
void copyFromBuffer_edges  (Kokkos::View<double***> view, Buffer *recv_buffer, int ivar);
void copyFromBuffer_corners(Kokkos::View<double***> view, Buffer *recv_buffer, int ivar);

void exchangeBuffer(Buffer *send_buffer, Buffer *recv_buffer, Grid *grid);
void exchangeBuffer_faces(Buffer *send_buffer, Buffer *recv_buffer, Grid *grid);
void exchangeBuffer_edges(Buffer *send_buffer, Buffer *recv_buffer, Grid *grid);
void exchangeBuffer_corners(Buffer *send_buffer, Buffer *recv_buffer, Grid *grid);

/**
 * @file buffer.cpp
 * buffer class implementation
 */
#include "buffer.hpp"

Buffer::Buffer(std::array<int, 3> const & ng, 
               std::array<int, 3> const & nx_local_ng, 
               int const nvar)
{    
    Nghost[0] = ng[0];
    Nghost[1] = ng[1];
    Nghost[2] = ng[2];

    for(int f=0; f<2; f++)
    {
        if(Nghost[0]) faceBuffer[0][f] = Kokkos::DualView<double****> ("", Nghost[0], nx_local_ng[1], nx_local_ng[2], nvar);
        if(Nghost[1]) faceBuffer[1][f] = Kokkos::DualView<double****> ("", nx_local_ng[0], Nghost[1], nx_local_ng[2], nvar);
        if(Nghost[2]) faceBuffer[2][f] = Kokkos::DualView<double****> ("", nx_local_ng[0], nx_local_ng[1], Nghost[2], nvar);
    }

    for(int e=0; e<4; e++)
    {
        if(Nghost[1] && Nghost[2]) edgeBuffer[0][e] = Kokkos::DualView<double****> ("edgeBuffer", nx_local_ng[0], Nghost[1], Nghost[2], nvar);
        if(Nghost[0] && Nghost[2]) edgeBuffer[1][e] = Kokkos::DualView<double****> ("edgeBuffer", Nghost[0], nx_local_ng[1], Nghost[2], nvar);
        if(Nghost[0] && Nghost[1]) edgeBuffer[2][e] = Kokkos::DualView<double****> ("edgeBuffer", Nghost[0], Nghost[1], nx_local_ng[2], nvar);
    }
    
    if(Nghost[0] && Nghost[1] && Nghost[2])
    {
        for (int z=0; z<2; z++)
        {
            for (int c=0; c<4; c++)
            {
                cornerBuffer[z][c] = Kokkos::DualView<double****> ("cornerBuffer", Nghost[0], Nghost[1], Nghost[2], nvar);
            }
        }
    }
    
    for (int i=0; i<3; i++)
    {
        faceBufferSize[i] = faceBuffer[i][0].h_view.size();
        edgeBufferSize[i] = edgeBuffer[i][0].h_view.size();
    }
    cornerBufferSize = Nghost[0]*Nghost[1]*Nghost[2]*nvar;
}
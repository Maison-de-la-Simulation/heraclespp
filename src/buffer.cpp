/**
 * @file buffer.cpp
 * buffer class implementation
 */
#include "buffer.hpp"

namespace novapp
{

Buffer::Buffer(std::array<int, 3> const & ng, 
               std::array<int, 3> const & nx_local_ng, 
               int const nvar)
{    
    Nghost[0] = ng[0];
    Nghost[1] = ng[1];
    Nghost[2] = ng[2];

    for(int f=0; f<2; f++)
    {
        if(Nghost[0]) faceBuffer[0][f] = KDV_double_4d ("", Nghost[0], nx_local_ng[1], nx_local_ng[2], nvar);
        if(Nghost[1]) faceBuffer[1][f] = KDV_double_4d ("", nx_local_ng[0], Nghost[1], nx_local_ng[2], nvar);
        if(Nghost[2]) faceBuffer[2][f] = KDV_double_4d ("", nx_local_ng[0], nx_local_ng[1], Nghost[2], nvar);
    }

    for(int e=0; e<4; e++)
    {
        if(Nghost[1] && Nghost[2]) edgeBuffer[0][e] = KDV_double_4d ("edgeBuffer", nx_local_ng[0], Nghost[1], Nghost[2], nvar);
        if(Nghost[0] && Nghost[2]) edgeBuffer[1][e] = KDV_double_4d ("edgeBuffer", Nghost[0], nx_local_ng[1], Nghost[2], nvar);
        if(Nghost[0] && Nghost[1]) edgeBuffer[2][e] = KDV_double_4d ("edgeBuffer", Nghost[0], Nghost[1], nx_local_ng[2], nvar);
    }
    
    if(Nghost[0] && Nghost[1] && Nghost[2])
    {
        for (int z=0; z<2; z++)
        {
            for (int c=0; c<4; c++)
            {
                cornerBuffer[z][c] = KDV_double_4d ("cornerBuffer", Nghost[0], Nghost[1], Nghost[2], nvar);
            }
        }
    }
}

} // namespace novapp

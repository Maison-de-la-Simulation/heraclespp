/**
 * @file buffer.cpp
 * buffer class implementation
 */
#include "buffer.hpp"

Buffer::Buffer(Grid *grid, int nvar)
{
    
    Nghost = grid->Nghost;
    
    for(int f=0; f<2; f++)
    {
        faceBuffer[0][f] = Kokkos::DualView<double****> ("", Nghost, grid->Nx_local_ng[1], grid->Nx_local_ng[2], nvar);
        faceBuffer[1][f] = Kokkos::DualView<double****> ("", grid->Nx_local_ng[0], Nghost, grid->Nx_local_ng[2], nvar);
        faceBuffer[2][f] = Kokkos::DualView<double****> ("", grid->Nx_local_ng[0], grid->Nx_local_ng[1], Nghost, nvar);
#if defined(Kokkos_ENABLE_DEBUG)
        Kokkos::deep_copy(faceBuffer[0][f].view_device(), NAN);
        Kokkos::deep_copy(faceBuffer[1][f].view_device(), NAN);
        Kokkos::deep_copy(faceBuffer[2][f].view_device(), NAN);
#endif
    }

    for(int j=0; j<2; j++)
    {
        for (int k = 0; k < 2; k++)
        {
            edgeBuffer[0][j][k] = Kokkos::DualView<double****> ("edgeBuffer", grid->Nx_local_ng[0], Nghost, Nghost, nvar);
            edgeBuffer[1][j][k] = Kokkos::DualView<double****> ("edgeBuffer", Nghost, grid->Nx_local_ng[1], Nghost, nvar);
            edgeBuffer[2][j][k] = Kokkos::DualView<double****> ("edgeBuffer", Nghost, Nghost, grid->Nx_local_ng[2], nvar);
#if defined(Kokkos_ENABLE_DEBUG)
            Kokkos::deep_copy(edgeBuffer[0][j][k].view_device(), NAN);
            Kokkos::deep_copy(edgeBuffer[1][j][k].view_device(), NAN);
            Kokkos::deep_copy(edgeBuffer[2][j][k].view_device(), NAN);
#endif
        }
    }
    
    for (int i=0; i<2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                cornerBuffer[i][j][k] = Kokkos::DualView<double****> ("cornerBuffer", Nghost, Nghost, Nghost, nvar);
#if defined(Kokkos_ENABLE_DEBUG)
                Kokkos::deep_copy(cornerBuffer[i][j][k].view_device(), NAN);
#endif
            }
        }
    }
    
    for (int i=0; i<3; i++)
    {
        faceBufferSize[i] = faceBuffer[i][0].extent(0)*faceBuffer[i][0].extent(1)*faceBuffer[i][0].extent(2)*faceBuffer[i][0].extent(3);
        edgeBufferSize[i] = edgeBuffer[i][0][0].extent(0)*edgeBuffer[i][0][0].extent(1)*edgeBuffer[i][0][0].extent(2)*edgeBuffer[i][0][0].extent(3);
    }
    cornerBufferSize = Nghost*Nghost*Nghost*nvar;
};

void copyToBuffer(Kokkos::View<double***> view, Buffer *buffer, int ivar)
{
    copyToBuffer_faces(view, buffer, ivar);  
    copyToBuffer_edges(view, buffer, ivar);
    copyToBuffer_corners(view, buffer, ivar);
    Kokkos::fence();
};

void copyFromBuffer(Kokkos::View<double***> view, Buffer *buffer, int ivar)
{
    copyFromBuffer_faces(view, buffer, ivar);  
    copyFromBuffer_edges(view, buffer, ivar);
    copyFromBuffer_corners(view, buffer, ivar);
    Kokkos::fence();
};

void exchangeBuffer(Buffer *send_buffer, Buffer *recv_buffer, Grid *grid)
{    
    exchangeBuffer_faces(send_buffer, recv_buffer, grid);
    exchangeBuffer_edges(send_buffer, recv_buffer, grid);
    exchangeBuffer_corners(send_buffer, recv_buffer, grid);
};


void copyToBuffer_corners(Kokkos::View<double***> view, Buffer *buffer, int ivar)
{
    int ng = buffer->Nghost;
    int lim0 = view.extent(0)-ng;
    int lim1 = view.extent(1)-ng;
    int lim2 = view.extent(2)-ng;
    
    for(int i=0; i<2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                auto sub_view = Kokkos::subview(view, Kokkos::pair(i*lim0, i*lim0+ng),
                                                      Kokkos::pair(j*lim1, j*lim1+ng),
                                                      Kokkos::pair(k*lim2, k*lim2+ng));

                auto sub_view_buffer = Kokkos::subview(buffer->cornerBuffer[i][j][k].view_device(), 
                                                       Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, ivar);
                                                                            
                Kokkos::deep_copy(sub_view_buffer, sub_view);
            }   
        }   
    }
};

void copyToBuffer_edges(Kokkos::View<double***> view, Buffer *buffer, int ivar)
{
    int ng = buffer->Nghost;
    int lim0 = view.extent(0)-ng;
    int lim1 = view.extent(1)-ng;
    int lim2 = view.extent(2)-ng;

    for(int j=0; j<2; j++)
    {
        for(int k=0; k<2; k++)
        {
            auto sub_view = Kokkos::subview(view,Kokkos::pair(ng, lim0),
                                             Kokkos::pair(j*lim1, ng+j*lim1),
                                             Kokkos::pair(k*lim2, ng+k*lim2));
            auto sub_buffer = Kokkos::subview(buffer->edgeBuffer[0][j][k].view_device(), 
                                           Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, ivar);                      
            Kokkos::deep_copy(sub_buffer, sub_view);
        }
    }

    for(int j=0; j<2; j++)
    {
        for(int k=0; k<2; k++)
        {
            auto sub_view = Kokkos::subview(view,Kokkos::pair(j*lim0, ng+j*lim0),
                                             Kokkos::pair(ng, lim1),
                                             Kokkos::pair(k*lim2, ng+k*lim2));
            auto sub_buffer = Kokkos::subview(buffer->edgeBuffer[1][j][k].view_device(), 
                                           Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, ivar);                      
            Kokkos::deep_copy(sub_buffer, sub_view);
        }
    }

    for(int j=0; j<2; j++)
    {
        for(int k=0; k<2; k++)
        {
            auto sub_view = Kokkos::subview(view,Kokkos::pair(j*lim0, ng+j*lim0),
                                             Kokkos::pair(k*lim1, ng+k*lim1),
                                             Kokkos::pair(ng, lim2));
            auto sub_buffer = Kokkos::subview(buffer->edgeBuffer[2][j][k].view_device(), 
                                           Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, ivar);                      
            Kokkos::deep_copy(sub_buffer, sub_view);
        }
    }
};

void copyToBuffer_faces(Kokkos::View<double***> view, Buffer *buffer, int ivar)
{
    int ng = buffer->Nghost;
    int lim0 = view.extent(0)-ng;
    int lim1 = view.extent(1)-ng;
    int lim2 = view.extent(2)-ng;

    for(int i=0; i<3; i++)
    {
        int Istart = ng, Ystart=ng, Zstart=ng;
        int Iend = lim0, Yend=lim1, Zend=lim2;
        for(int f=0; f<2; f++)
        {          
            if(i==0) { Istart = f*lim0 ; Iend = f*lim0+ng; }
            if(i==1) { Ystart = f*lim1 ; Yend = f*lim1+ng; }
            if(i==2) { Zstart = f*lim2 ; Zend = f*lim2+ng; }

            auto sub_view = Kokkos::subview(view, Kokkos::pair(Istart, Iend),
                                            Kokkos::pair(Ystart, Yend),
                                            Kokkos::pair(Zstart, Zend));

            auto sub_view_buffer = Kokkos::subview(buffer->faceBuffer[i][f].view_device(), 
                                                   Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, ivar);
                                                                            
            Kokkos::deep_copy(sub_view_buffer, sub_view);
        }
    }
};

void copyFromBuffer_corners(Kokkos::View<double***> view, Buffer *buffer, int ivar)
{
    int ng = buffer->Nghost;
    int lim0 = view.extent(0)-ng;
    int lim1 = view.extent(1)-ng;
    int lim2 = view.extent(2)-ng;

    for(int i=0; i<2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                auto sub_view = Kokkos::subview(view, Kokkos::pair(i*lim0, i*lim0+ng),
                                                      Kokkos::pair(j*lim1, j*lim1+ng),
                                                      Kokkos::pair(k*lim2, k*lim2+ng));

                auto sub_view_buffer = Kokkos::subview(buffer->cornerBuffer[i][j][k].view_device(), 
                                                       Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, ivar);
                                                                            
                Kokkos::deep_copy(sub_view, sub_view_buffer);
            }   
        }   
    }
};

void copyFromBuffer_edges(Kokkos::View<double***> view, Buffer *buffer, int ivar)
{
    int ng = buffer->Nghost;
    int lim0 = view.extent(0)-ng;
    int lim1 = view.extent(1)-ng;
    int lim2 = view.extent(2)-ng;

    for(int j=0; j<2; j++)
    {
        for(int k=0; k<2; k++)
        {
            auto sub_view = Kokkos::subview(view,Kokkos::pair(ng, lim0),
                                             Kokkos::pair(j*lim1, ng+j*lim1),
                                             Kokkos::pair(k*lim2, ng+k*lim2));
            auto sub_buffer = Kokkos::subview(buffer->edgeBuffer[0][j][k].view_device(), 
                                           Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, ivar);                      
            Kokkos::deep_copy(sub_view, sub_buffer);
        }
    }

    for(int j=0; j<2; j++)
    {
        for(int k=0; k<2; k++)
        {
            auto sub_view = Kokkos::subview(view,Kokkos::pair(j*lim0, ng+j*lim0),
                                             Kokkos::pair(ng, lim1),
                                             Kokkos::pair(k*lim2, ng+k*lim2));
            auto sub_buffer = Kokkos::subview(buffer->edgeBuffer[1][j][k].view_device(), 
                                           Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, ivar);                      
            Kokkos::deep_copy(sub_view, sub_buffer);
        }
    }

    for(int j=0; j<2; j++)
    {
        for(int k=0; k<2; k++)
        {
            auto sub_view = Kokkos::subview(view,Kokkos::pair(j*lim0, ng+j*lim0),
                                             Kokkos::pair(k*lim1, ng+k*lim1),
                                             Kokkos::pair(ng, lim2));
            auto sub_buffer = Kokkos::subview(buffer->edgeBuffer[2][j][k].view_device(), 
                                           Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, ivar);                      
            Kokkos::deep_copy(sub_view, sub_buffer);
        }
    }
};

void copyFromBuffer_faces(Kokkos::View<double***> view, Buffer *buffer, int ivar)
{
    int ng = buffer->Nghost;
    int lim0 = view.extent(0)-ng;
    int lim1 = view.extent(1)-ng;
    int lim2 = view.extent(2)-ng;

    for(int i=0; i<3; i++)
    {
        int Istart = ng, Ystart=ng, Zstart=ng;
        int Iend = lim0, Yend=lim1, Zend=lim2;
        for(int f=0; f<2; f++)
        {          
            if(i==0) { Istart = f*lim0 ; Iend = f*lim0+ng; }
            if(i==1) { Ystart = f*lim1 ; Yend = f*lim1+ng; }
            if(i==2) { Zstart = f*lim2 ; Zend = f*lim2+ng; }

            auto sub_view = Kokkos::subview(view, Kokkos::pair(Istart, Iend),
                                            Kokkos::pair(Ystart, Yend),
                                            Kokkos::pair(Zstart, Zend));

            auto sub_view_buffer = Kokkos::subview(buffer->faceBuffer[i][f].view_device(), 
                                                   Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, ivar);
                                                                            
            Kokkos::deep_copy(sub_view, sub_view_buffer);
        }
    }
};

void exchangeBuffer_faces(Buffer *send_buffer, Buffer *recv_buffer, Grid *grid)
{    
    MPI_Status mpi_status;
    
    int left_neighbor, right_neighbor;
    for(int i=0; i<3; i++)
    {
        MPI_Cart_shift(grid->comm_cart, i, -1, &right_neighbor, &left_neighbor);
        //send to left, recv from right
        MPI_Sendrecv(send_buffer->faceBuffer[i][0].view_device().data(), 
                     send_buffer->faceBufferSize[i], MPI_DOUBLE,
                     left_neighbor, i,
                     recv_buffer->faceBuffer[i][1].view_device().data(),
                     recv_buffer->faceBufferSize[i], MPI_DOUBLE,
                     right_neighbor, i,
                     MPI_COMM_WORLD, &mpi_status);

        //send to right, recv from left
        MPI_Sendrecv(send_buffer->faceBuffer[i][1].view_device().data(), 
                     send_buffer->faceBufferSize[i], MPI_DOUBLE,
                     right_neighbor, i,
                     recv_buffer->faceBuffer[i][0].view_device().data(),
                     recv_buffer->faceBufferSize[i], MPI_DOUBLE,
                     left_neighbor, i,
                     MPI_COMM_WORLD, &mpi_status);
    }
};

void exchangeBuffer_edges(Buffer *send_buffer, Buffer *recv_buffer, Grid *grid)
{    
    MPI_Status mpi_status;
    
    int left_neighbor, right_neighbor;
    for(int i=0; i<3; i++)
    {
        for(int f=0; f<4; f++)
        {
            left_neighbor = grid->NeighborRank[i==0?1:(f%2==0?0:2)][i==1?1:(i==1?(f%2==0?0:2):(f/2>0?2:0))][i==2?1:(f/2>0?2:0)];
            right_neighbor = grid->NeighborRank[i==0?1:(f%2==0?2:0)][i==1?1:(i==1?(f%2==0?0:2):(f/2>0?0:2))][i==2?1:(f/2>0?0:2)];
            MPI_Sendrecv(send_buffer->edgeBuffer[i][f/2][f%2].view_device().data(), 
                         send_buffer->edgeBufferSize[i], MPI_DOUBLE,
                         left_neighbor, 88,
                         recv_buffer->edgeBuffer[i][(3-f)/2][(3-f)%2].view_device().data(),
                         recv_buffer->edgeBufferSize[i], MPI_DOUBLE,
                         right_neighbor, 88,
                         MPI_COMM_WORLD, &mpi_status);
        }
    }  
};

void exchangeBuffer_corners(Buffer *send_buffer, Buffer *recv_buffer, Grid *grid)
{    
    MPI_Status mpi_status;
    
    int left_neighbor, right_neighbor; 
    for(int j=0; j<2; j++)
    {
        for (int i=0; i<2; i++)
        {
            left_neighbor = grid->NeighborRank[i*2][j*2][0];
            right_neighbor = grid->NeighborRank[2-2*i][2-2*j][2];       
            MPI_Sendrecv(send_buffer->cornerBuffer[i][j][0].view_device().data(), 
                         send_buffer->cornerBufferSize, MPI_DOUBLE,
                         left_neighbor, 77,
                         recv_buffer->cornerBuffer[(i+1)%2][(j+1)%2][1].view_device().data(),
                         recv_buffer->cornerBufferSize, MPI_DOUBLE,
                         right_neighbor, 77,
                         MPI_COMM_WORLD, &mpi_status);

            left_neighbor = grid->NeighborRank[i*2][j*2][2];
            right_neighbor = grid->NeighborRank[2-2*i][2-2*j][0];
            MPI_Sendrecv(send_buffer->cornerBuffer[i][j][1].view_device().data(), 
                         send_buffer->cornerBufferSize, MPI_DOUBLE,
                         left_neighbor, 77,
                         recv_buffer->cornerBuffer[(i+1)%2][(j+1)%2][0].view_device().data(),
                         recv_buffer->cornerBufferSize, MPI_DOUBLE,
                         right_neighbor, 77,
                         MPI_COMM_WORLD, &mpi_status);
        }
    }
};
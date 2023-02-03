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
    }

    for(int j=0; j<2; j++)
    {
        for (int k = 0; k < 2; k++)
        {
            edgeBuffer[0][j][k] = Kokkos::DualView<double****> ("edgeBuffer", grid->Nx_local_ng[0], Nghost, Nghost, nvar);
            edgeBuffer[1][j][k] = Kokkos::DualView<double****> ("edgeBuffer", Nghost, grid->Nx_local_ng[1], Nghost, nvar);
            edgeBuffer[2][j][k] = Kokkos::DualView<double****> ("edgeBuffer", Nghost, Nghost, grid->Nx_local_ng[2], nvar);
        }
        
    }
    
    for (int i=0; i<2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                cornerBuffer[i][j][k] = Kokkos::DualView<double****> ("cornerBuffer", Nghost, Nghost, Nghost, nvar);
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
                Kokkos::parallel_for("copyToBuffer_corners", 
                                     Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{ng,ng,ng}),
                                     KOKKOS_LAMBDA (int ii, int jj, int kk) {
                                        buffer->cornerBuffer[i][j][k].view_device()(ii, jj, kk, ivar) 
                                      = view(ii+i*lim0,jj+j*lim1,kk+k*lim2);
                                    });
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

    Kokkos::parallel_for("copyToBuffer_edge_0123", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({ng,0,0},{lim0,ng,ng}),
                          KOKKOS_LAMBDA (int i, int j, int k) 
    {
        buffer->edgeBuffer[0][0][0].view_device()(i-ng, j, k, ivar) = view(i,j,     k);
        buffer->edgeBuffer[0][1][0].view_device()(i-ng, j, k, ivar) = view(i,j+lim1,k);
        buffer->edgeBuffer[0][0][1].view_device()(i-ng, j, k, ivar) = view(i,j,     k+lim2);
        buffer->edgeBuffer[0][1][1].view_device()(i-ng, j, k, ivar) = view(i,j+lim1,k+lim2);
    });

    Kokkos::parallel_for("copyToBuffer_edge_4567", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,ng,0},{ng,lim1,ng}),
                          KOKKOS_LAMBDA (int i, int j, int k) 
    {
        buffer->edgeBuffer[1][0][0].view_device()(i, j-ng, k, ivar) = view(i,     j,k);
        buffer->edgeBuffer[1][1][0].view_device()(i, j-ng, k, ivar) = view(i+lim0,j,k);
        buffer->edgeBuffer[1][0][1].view_device()(i, j-ng, k, ivar) = view(i,     j,k+lim2);
        buffer->edgeBuffer[1][1][1].view_device()(i, j-ng, k, ivar) = view(i+lim0,j,k+lim2);
    });

    Kokkos::parallel_for("copyToBuffer_edge_891011", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,ng},{ng,ng,lim2}),
                          KOKKOS_LAMBDA (int i, int j, int k) 
    {
        buffer->edgeBuffer[2][0][0].view_device()(i, j, k-ng, ivar) = view(i,     j,k);
        buffer->edgeBuffer[2][1][0].view_device()(i, j, k-ng, ivar) = view(i+lim0,j,k);
        buffer->edgeBuffer[2][0][1].view_device()(i, j, k-ng, ivar) = view(i,     j+lim1,k);
        buffer->edgeBuffer[2][1][1].view_device()(i, j, k-ng, ivar) = view(i+lim0,j+lim1,k);
    });

};

void copyToBuffer_faces(Kokkos::View<double***> view, Buffer *buffer, int ivar)
{
    int ng = buffer->Nghost;
    int lim0 = view.extent(0)-ng;
    int lim1 = view.extent(1)-ng;
    int lim2 = view.extent(2)-ng;

    for(int f=0; f<2; f++)
    {
        Kokkos::parallel_for("copyToBuffer_X", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({f==0?0:lim0,ng,ng},{f==0?ng:lim0+ng,lim1,lim2}),        
                             KOKKOS_LAMBDA (int i, int j, int k) {
                                 buffer->faceBuffer[0][f].view_device()(i-f*lim0, j-ng, k-ng, ivar) = view(i,j,k);
                            });

        Kokkos::parallel_for("copyToBuffer_Y", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({ng,f==0?0:lim1,ng},{lim0,f==0?ng:lim1+ng,lim2}),        
                             KOKKOS_LAMBDA (int i, int j, int k) {
                                 buffer->faceBuffer[1][f].view_device()(i-ng, j-f*lim1, k-ng, ivar) = view(i,j,k);
                            });
        Kokkos::parallel_for("copyToBuffer_Z", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({ng,ng, f==0?0:lim2},{lim0,lim1,f==0?ng:lim2+ng}),        
                             KOKKOS_LAMBDA (int i, int j, int k) {
                                 buffer->faceBuffer[2][f].view_device()(i-ng, j-ng, k-f*lim2, ivar) = view(i,j,k);
                            });
    }
};

void copyFromBuffer(Kokkos::View<double***> view, Buffer *buffer, int ivar)
{
    copyFromBuffer_faces(view, buffer, ivar);  
    copyFromBuffer_edges(view, buffer, ivar);
    copyFromBuffer_corners(view, buffer, ivar);
    Kokkos::fence();
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
                Kokkos::parallel_for("copyFromBuffer_corners", 
                                     Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{ng,ng,ng}),
                                     KOKKOS_LAMBDA (int ii, int jj, int kk) {
                                        view(ii+i*lim0,jj+j*lim1,kk+k*lim2)
                                      = buffer->cornerBuffer[i][j][k].view_device()(ii, jj, kk, ivar);
                                    });
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

    Kokkos::parallel_for("copyFromBuffer_edge_0123", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({ng,0,0},{lim0,ng,ng}),
                          KOKKOS_LAMBDA (int i, int j, int k) 
    {
        view(i,j,     k)      = buffer->edgeBuffer[0][0][0].view_device()(i-ng, j, k, ivar) ;
        view(i,j+lim1,k)      = buffer->edgeBuffer[0][1][0].view_device()(i-ng, j, k, ivar) ;
        view(i,j,     k+lim2) = buffer->edgeBuffer[0][0][1].view_device()(i-ng, j, k, ivar) ;
        view(i,j+lim1,k+lim2) = buffer->edgeBuffer[0][1][1].view_device()(i-ng, j, k, ivar) ;
    });

    Kokkos::parallel_for("copyFromBuffer_edge_4567", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,ng,0},{ng,lim1,ng}),
                          KOKKOS_LAMBDA (int i, int j, int k) 
    {
        view(i,     j,k)      = buffer->edgeBuffer[1][0][0].view_device()(i, j-ng, k, ivar) ;
        view(i+lim0,j,k)      = buffer->edgeBuffer[1][1][0].view_device()(i, j-ng, k, ivar) ;
        view(i,     j,k+lim2) = buffer->edgeBuffer[1][0][1].view_device()(i, j-ng, k, ivar) ;
        view(i+lim0,j,k+lim2) = buffer->edgeBuffer[1][1][1].view_device()(i, j-ng, k, ivar) ;
    });

    Kokkos::parallel_for("copyFromBuffer_edge_891011", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,ng},{ng,ng,lim2}),
                          KOKKOS_LAMBDA (int i, int j, int k) 
    {
        view(i,     j,k)      = buffer->edgeBuffer[2][0][0].view_device()(i, j, k-ng, ivar) ;
        view(i+lim0,j,k)      = buffer->edgeBuffer[2][1][0].view_device()(i, j, k-ng, ivar) ;
        view(i,     j+lim1,k) = buffer->edgeBuffer[2][0][1].view_device()(i, j, k-ng, ivar) ;
        view(i+lim0,j+lim1,k) = buffer->edgeBuffer[2][1][1].view_device()(i, j, k-ng, ivar) ;
    });
};

void copyFromBuffer_faces(Kokkos::View<double***> view, Buffer *buffer, int ivar)
{
    int ng = buffer->Nghost;
    int lim0 = view.extent(0)-ng;
    int lim1 = view.extent(1)-ng;
    int lim2 = view.extent(2)-ng;
    
    for(int f=0; f<2; f++)
    {
        Kokkos::parallel_for("copyToBuffer_X", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({f==0?0:lim0,ng,ng},{f==0?ng:lim0+ng,lim1,lim2}),        
                             KOKKOS_LAMBDA (int i, int j, int k) {
                                 view(i,j,k) = buffer->faceBuffer[0][f].view_device()(i-f*lim0, j-ng, k-ng, ivar) ;
                            });

        Kokkos::parallel_for("copyToBuffer_Y", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({ng,f==0?0:lim1,ng},{lim0,f==0?ng:lim1+ng,lim2}),        
                             KOKKOS_LAMBDA (int i, int j, int k) {
                                 view(i,j,k) = buffer->faceBuffer[1][f].view_device()(i-ng, j-f*lim1, k-ng, ivar) ;
                            });
        Kokkos::parallel_for("copyToBuffer_Z", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({ng,ng, f==0?0:lim2},{lim0,lim1,f==0?ng:lim2+ng}),        
                             KOKKOS_LAMBDA (int i, int j, int k) {
                                 view(i,j,k) = buffer->faceBuffer[2][f].view_device()(i-ng, j-ng, k-f*lim2, ivar) ;
                            });
    }
};


void exchangeBuffer(Buffer *send_buffer, Buffer *recv_buffer, Grid *grid)
{
    std::cout<<"============== exchange buffer started"<<std::endl;
    
    std::cout<<"== face buffer (6)"<<std::endl;
    MPI_Status mpi_status;
    
    // int src_rank, dest_rank;
    int left_neighbor, right_neighbor;
    for(int i=0; i<1; i++)
    {
        MPI_Cart_shift(grid->comm_cart, i, -1, &right_neighbor, &left_neighbor);
        //send to left, recv from right
        // std::cout<<"LEFT: proc "<<grid->mpi_rank<<" send to "<<left_neighbor<<" and recv from "<<right_neighbor<<std::endl;
        MPI_Sendrecv(send_buffer->faceBuffer[i][0].view_device().data(), 
                     send_buffer->faceBufferSize[i], MPI_DOUBLE,
                     left_neighbor, i,
                     recv_buffer->faceBuffer[i][1].view_device().data(),
                     recv_buffer->faceBufferSize[i], MPI_DOUBLE,
                     right_neighbor, i,
                     MPI_COMM_WORLD, &mpi_status);
        //send to right, recv from left
        // std::cout<<"RIGHT: proc "<<grid->mpi_rank<<" send to "<<right_neighbor<<" and recv from "<<left_neighbor<<std::endl;
        MPI_Sendrecv(send_buffer->faceBuffer[i][1].view_device().data(), 
                     send_buffer->faceBufferSize[i], MPI_DOUBLE,
                     right_neighbor, i,
                     recv_buffer->faceBuffer[i][0].view_device().data(),
                     recv_buffer->faceBufferSize[i], MPI_DOUBLE,
                     left_neighbor, i,
                     MPI_COMM_WORLD, &mpi_status);

    }

    std::cout<<"== edge buffer (12)"<<std::endl;
    for(int i=0; i<3; i++)
    {
        for(int f=0; f<4; f++)
        {
            left_neighbor = grid->NeighborRank[i==0?1:(f%2==0?0:2)][i==1?1:(i==1?(f%2==0?0:2):(f/2>0?2:0))][i==2?1:(f/2>0?2:0)];
            right_neighbor = grid->NeighborRank[i==0?1:(f%2==0?2:0)][i==1?1:(i==1?(f%2==0?0:2):(f/2>0?0:2))][i==2?1:(f/2>0?0:2)];
            // std::cout<<"proc "<<grid->mpi_rank<<" send edge "<<i*4+f/2*2+f%2<<" to rank "<<left_neighbor<<std::endl;
            // std::cout<<"proc "<<grid->mpi_rank<<" recv edge "<<i*4+(3-f)/2*2+(3-f)%2<<" to rank "<<right_neighbor<<std::endl;
            MPI_Sendrecv(send_buffer->edgeBuffer[i][f/2][f%2].view_device().data(), 
                         send_buffer->edgeBufferSize[i], MPI_DOUBLE,
                         left_neighbor, 88,
                         recv_buffer->edgeBuffer[i][(3-f)/2][(3-f)%2].view_device().data(),
                         recv_buffer->edgeBufferSize[i], MPI_DOUBLE,
                         right_neighbor, 88,
                         MPI_COMM_WORLD, &mpi_status);
        }
    }
    
    std::cout<<"== corner buffer (8)"<<std::endl;
    
    for(int j=0; j<2; j++)
    {
        for (int i=0; i<2; i++)
        {
            left_neighbor = grid->NeighborRank[i*2][j*2][0];
            right_neighbor = grid->NeighborRank[2-2*i][2-2*j][2];
            // std::cout<<"proc "<<grid->mpi_rank<<" send corner "<<i+j*2<<" to rank "<<left_neighbor<<std::endl;
            // std::cout<<"proc "<<grid->mpi_rank<<" recv corner "<<((i+1)%2)+((j+1)%2)*2+4<<" from rank "<<right_neighbor<<std::endl;
            
            MPI_Sendrecv(send_buffer->cornerBuffer[i][j][0].view_device().data(), 
                         send_buffer->cornerBufferSize, MPI_DOUBLE,
                         left_neighbor, 77,
                         recv_buffer->cornerBuffer[(i+1)%2][(j+1)%2][1].view_device().data(),
                         recv_buffer->cornerBufferSize, MPI_DOUBLE,
                         right_neighbor, 77,
                         MPI_COMM_WORLD, &mpi_status);

            left_neighbor = grid->NeighborRank[i*2][j*2][2];
            right_neighbor = grid->NeighborRank[2-2*i][2-2*j][0];
            // std::cout<<"proc "<<grid->mpi_rank<<" send corner "<<i+j*2+4<<" to rank "<<left_neighbor<<std::endl;
            // std::cout<<"proc "<<grid->mpi_rank<<" recv corner "<<((i+1)%2)+((j+1)%2)*2<<" from rank "<<right_neighbor<<std::endl;
            MPI_Sendrecv(send_buffer->cornerBuffer[i][j][1].view_device().data(), 
                         send_buffer->cornerBufferSize, MPI_DOUBLE,
                         left_neighbor, 77,
                         recv_buffer->cornerBuffer[(i+1)%2][(j+1)%2][0].view_device().data(),
                         recv_buffer->cornerBufferSize, MPI_DOUBLE,
                         right_neighbor, 77,
                         MPI_COMM_WORLD, &mpi_status);
        }
    }
    
    std::cout<<"============== exchange buffer finished"<<std::endl;
};


void setView(Kokkos::View<double***> view, double value)
{
    Kokkos::parallel_for("setView", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{view.extent(0),view.extent(1),view.extent(2)}),
                          KOKKOS_LAMBDA (int i, int j, int k) 
    {
        view(i,j,k) = value;
    });
};

void printView(Kokkos::View<double***> view)
{
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    if(mpi_rank) return;
    Kokkos::parallel_for("printView", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{view.extent(0),view.extent(1)}),
                          KOKKOS_LAMBDA (int i, int j) 
    {
        std::cout<<"view("<<i<<","<<j<<") = "<<view(i,j,0);
        std::cout<<" "<<view(i,j,1);
        std::cout<<" "<<view(i,j,2);
        std::cout<<" "<<view(i,j,3);
        std::cout<<" "<<view(i,j,4);
        std::cout<<" "<<view(i,j,5)<<std::endl;
    });
};
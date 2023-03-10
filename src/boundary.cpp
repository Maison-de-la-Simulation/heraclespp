/**
 * @file exchange.cpp
 * Holo exchange implementation
 */
#include <array>
#include <mpi.h>
#include "boundary.hpp"

namespace novapp
{

namespace {

    template<std::size_t n, size_t m>
    void modifyDtoH(std::array<std::array<KDV_double_4d,n>,m> & buffer_element);   

    template<std::size_t n, size_t m>
    void modifyHtoD(std::array<std::array<KDV_double_4d,n>,m> & buffer_element);   

    //! Update buffer from device data before border exchange
    //! @param[in] view Variable field device data
    //! @param[in] send_buffer Buffer used to store and later send the border value
    //! @param[in] ivar Index of input variable
    void copyToBuffer(KV_cdouble_3d const view, Buffer & send_buffer, int const ivar);   
    

    //! Update face buffer from device data before border exchange
    //! @param[in] view Variable field device data
    //! @param[in] send_buffer Buffer used to store and later send the border value
    //! @param[in] ivar Index of input variable
    void copyToBuffer_faces(KV_cdouble_3d const view, Buffer & send_buffer, int const ivar); 
    

    //! Update edge buffer from device data before border exchange
    //! @param[in] view Variable field device data
    //! @param[in] send_buffer Buffer used to store and later send the border value
    //! @param[in] ivar Index of input variable
    void copyToBuffer_edges(KV_cdouble_3d const view, Buffer & send_buffer, int const ivar); 
    

    //! Update corner buffer from device data before border exchange
    //! @param[in] view Variable field device data
    //! @param[in] send_buffer Buffer used to store and later send the border value
    //! @param[in] ivar Index of input variable
    void copyToBuffer_corners(KV_cdouble_3d const view, Buffer & send_buffer, int const ivar);   
    

    //! Update device data from buffer after border exchange
    //! @param[in] view Variable field device data
    //! @param[in] recv_buffer Buffer used to receive border value
    //! @param[in] ivar index of variable field
    void copyFromBuffer(KV_double_3d const view, Buffer & recv_buffer, int const ivar); 
    

    //! Update device data from face buffer after border exchange
    //! @param[in] view Variable field device data
    //! @param[in] recv_buffer Buffer used to receive border value
    //! @param[in] ivar index of variable field
    void copyFromBuffer_faces(KV_double_3d const view, Buffer const & recv_buffer, int const ivar);   
    

    //! Update device data from edge buffer after border exchange
    //! @param[in] view Variable field device data
    //! @param[in] recv_buffer Buffer used to receive border value
    //! @param[in] ivar index of variable field
    void copyFromBuffer_edges(KV_double_3d const view, Buffer const & recv_buffer, int const ivar);   
    

    //! Update device data from corner buffer after border exchange
    //! @param[in] view Variable field device data
    //! @param[in] recv_buffer Buffer used to receive border value
    //! @param[in] ivar index of variable field
    void copyFromBuffer_corners(KV_double_3d const view, Buffer const & recv_buffer, int const ivar); 
    

    //! Performe the border value exchange
    //! @param[in] send_buffer Buffer used to send border value
    //! @param[in] recv_buffer Buffer used to receive border value
    //! @param[in] grid Grid of simulation.
    void exchangeBuffer(Buffer & send_buffer, Buffer & recv_buffer, std::array<int, 3> const & ng, MPI_Comm const & comm_cart, std::array<std::array<std::array<int, 3>, 3>, 3> const & nrank);    
    

    //! Performe the border face value exchange
    //! @param[in] send_buffer Buffer used to send border value
    //! @param[in] recv_buffer Buffer used to receive border value
    //! @param[in] grid Grid of simulation.
    void exchangeBuffer_faces(Buffer & send_buffer, Buffer & recv_buffer, std::array<int, 3> const & ng, MPI_Comm const & comm_cart);  
    

    //! Performe the border ege value exchange
    //! @param[in] send_buffer Buffer used to send border value
    //! @param[in] recv_buffer Buffer used to receive border value
    //! @param[in] grid Grid of simulation.
    void exchangeBuffer_edges(Buffer & send_buffer, Buffer & recv_buffer, std::array<int, 3> const & ng, MPI_Comm const & comm_cart, std::array<std::array<std::array<int, 3>, 3>, 3> const & nrank);  
    

    //! Performe the border corner value exchange
    //! @param[in] send_buffer Buffer used to send border value
    //! @param[in] recv_buffer Buffer used to receive border value
    //! @param[in] grid Grid of simulation.
    void exchangeBuffer_corners(Buffer & send_buffer, Buffer & recv_buffer, std::array<int, 3> const & ng, MPI_Comm const & comm_cart, std::array<std::array<std::array<int, 3>, 3>, 3> const & nrank);    
    
    void memcopy_VtoB(KV_cdouble_3d const view,
                      Kokkos::pair<int, int> lim0,
                      Kokkos::pair<int, int> lim1,
                      Kokkos::pair<int, int> lim2,
                      KV_double_4d buffer_type_element,
                      int ivar)
    {        
        Kokkos::deep_copy(Kokkos::subview(buffer_type_element, ALL, ALL, ALL, ivar),
                  Kokkos::subview(view, lim0, lim1, lim2));
    }

    void memcopy_BtoV(KV_double_3d const view,
                      Kokkos::pair<int, int> lim0,
                      Kokkos::pair<int, int> lim1,
                      Kokkos::pair<int, int> lim2,
                      KV_double_4d buffer_type_element,
                      int ivar)
    {        
        Kokkos::deep_copy(Kokkos::subview(view, lim0, lim1, lim2),
                  Kokkos::subview(buffer_type_element, ALL, ALL, ALL, ivar));
    }

    void copyToBuffer_corners(KV_cdouble_3d const view, Buffer & buffer, int const ivar) 
    {
        std::array<int, 3> ng = buffer.Nghost;
 
        if(ng[0] || ng[1] || ng[2]) { return; }
 
        int lim0 = view.extent(0)-2*ng[0];
        int lim1 = view.extent(1)-2*ng[1];
        int lim2 = view.extent(2)-2*ng[2];
        
        //corner(0,0,0)
        memcopy_VtoB(view, Kokkos::pair(ng[0], 2*ng[0]), 
                           Kokkos::pair(ng[1], 2*ng[1]),
                           Kokkos::pair(ng[2], 2*ng[2]), 
                           buffer.cornerBuffer[0][0].d_view, ivar);

        //corner(1,0,0)
        memcopy_VtoB(view, Kokkos::pair(lim0, lim0+ng[0]), 
                           Kokkos::pair(ng[1], 2*ng[1]),
                           Kokkos::pair(ng[2], 2*ng[2]), 
                           buffer.cornerBuffer[0][1].d_view, ivar);
        
        //corner(0,1,0)
        memcopy_VtoB(view, Kokkos::pair(ng[0], 2*ng[0]), 
                           Kokkos::pair(lim1, lim1+ng[1]),
                           Kokkos::pair(ng[2], 2*ng[2]), 
                           buffer.cornerBuffer[0][2].d_view, ivar);

        //corner(1,1,0)
        memcopy_VtoB(view, Kokkos::pair(lim0, lim0+ng[0]), 
                           Kokkos::pair(lim1, lim1+ng[1]),
                           Kokkos::pair(ng[2], 2*ng[2]), 
                           buffer.cornerBuffer[0][3].d_view, ivar);

        //corner(0,0,1)
        memcopy_VtoB(view, Kokkos::pair(ng[0], 2*ng[0]), 
                           Kokkos::pair(ng[1], 2*ng[1]),
                           Kokkos::pair(lim2, lim2+ng[2]), 
                           buffer.cornerBuffer[1][0].d_view, ivar);

        //corner(1,0,1)
        memcopy_VtoB(view, Kokkos::pair(lim0, lim0+ng[0]), 
                           Kokkos::pair(ng[1], 2*ng[1]),
                           Kokkos::pair(lim2, lim2+ng[2]), 
                           buffer.cornerBuffer[1][1].d_view, ivar);

        //corner(0,1,1)
        memcopy_VtoB(view, Kokkos::pair(ng[0], 2*ng[0]), 
                           Kokkos::pair(lim1, lim1+ng[1]),
                           Kokkos::pair(lim2, lim2+ng[2]), 
                           buffer.cornerBuffer[1][2].d_view, ivar);

        //corner(1,1,1)
        memcopy_VtoB(view, Kokkos::pair(lim0, lim0+ng[0]), 
                           Kokkos::pair(lim1, lim1+ng[1]),
                           Kokkos::pair(lim2, lim2+ng[2]), 
                           buffer.cornerBuffer[1][3].d_view, ivar);
    }

    void copyToBuffer_edges(KV_cdouble_3d const view, Buffer & buffer, int const ivar)   
    {
        std::array<int, 3> ng = buffer.Nghost;

        int lim0 = view.extent(0)-2*ng[0];
        int lim1 = view.extent(1)-2*ng[1];
        int lim2 = view.extent(2)-2*ng[2];

        if(ng[1] && ng[2])
        {
            //edge[0][0] : corner(0,0,0)<->corner(1,0,0)
            memcopy_VtoB(view, Kokkos::pair(ng[0], lim0+ng[0]), 
                               Kokkos::pair(ng[1], 2*ng[1]),
                               Kokkos::pair(ng[2], 2*ng[2]), 
                               buffer.edgeBuffer[0][0].d_view, ivar);

            //edge[0][1] : corner(0,1,0)<->corner(1,1,0)
            memcopy_VtoB(view, Kokkos::pair(ng[0], lim0+ng[0]), 
                               Kokkos::pair(lim1, lim1+ng[1]),
                               Kokkos::pair(ng[2], 2*ng[2]), 
                               buffer.edgeBuffer[0][1].d_view, ivar);

            //edge[0][2] : corner(0,0,1)<->corner(1,0,1)
            memcopy_VtoB(view, Kokkos::pair(ng[0], lim0+ng[0]), 
                               Kokkos::pair(ng[1], 2*ng[1]),
                               Kokkos::pair(lim2, lim2+ng[2]), 
                               buffer.edgeBuffer[0][2].d_view, ivar);

            //edge[0][3] : corner(0,1,1)<->corner(1,1,1)
            memcopy_VtoB(view, Kokkos::pair(ng[0], lim0+ng[0]), 
                               Kokkos::pair(lim1, lim1+ng[1]),
                               Kokkos::pair(lim2, lim2+ng[2]), 
                               buffer.edgeBuffer[0][3].d_view, ivar);
        }

        if(ng[0] && ng[2])
        {
            //edge[1][0] : corner(0,0,0)<->corner(0,1,0)
            memcopy_VtoB(view, Kokkos::pair(ng[0], 2*ng[0]), 
                               Kokkos::pair(ng[1], lim1+ng[1]),
                               Kokkos::pair(ng[2], 2*ng[2]), 
                               buffer.edgeBuffer[1][0].d_view, ivar);

            //edge[1][1] : corner(1,0,0)<->corner(1,1,0)
            memcopy_VtoB(view, Kokkos::pair(lim0, lim0+ng[0]), 
                               Kokkos::pair(ng[1], lim1+ng[1]),
                               Kokkos::pair(ng[2], 2*ng[2]), 
                               buffer.edgeBuffer[1][1].d_view, ivar);

            //edge[1][2] : corner(0,0,1)<->corner(0,1,1)
            memcopy_VtoB(view, Kokkos::pair(ng[0], 2*ng[0]), 
                               Kokkos::pair(ng[1], lim1+ng[1]),
                               Kokkos::pair(lim2, lim2+ng[2]), 
                               buffer.edgeBuffer[1][2].d_view, ivar);

            //edge[1][3] : corner(1,0,1)<->corner(1,1,1)
            memcopy_VtoB(view, Kokkos::pair(lim0, lim0+ng[0]), 
                               Kokkos::pair(ng[1], lim1+ng[1]),
                               Kokkos::pair(lim2, lim2+ng[2]), 
                               buffer.edgeBuffer[1][3].d_view, ivar);
        }

        if(ng[0] && ng[1])
        {
            //edge[2][0] : corner(0,0,0)<->corner(0,0,1)
            memcopy_VtoB(view, Kokkos::pair(ng[0], 2*ng[0]), 
                               Kokkos::pair(ng[1], 2*ng[1]),
                               Kokkos::pair(ng[2], lim2+ng[2]), 
                               buffer.edgeBuffer[2][0].d_view, ivar);

            //edge[2][1] : corner(1,0,0)<->corner(1,0,1)
            memcopy_VtoB(view, Kokkos::pair(lim0, lim0+ng[0]), 
                               Kokkos::pair(ng[1], 2*ng[1]),
                               Kokkos::pair(ng[2], lim2+ng[2]), 
                               buffer.edgeBuffer[2][1].d_view, ivar);

            //edge[2][2] : corner(0,1,0)<->corner(0,1,1)
            memcopy_VtoB(view, Kokkos::pair(ng[0], 2*ng[0]), 
                               Kokkos::pair(lim1, lim1+ng[1]),
                               Kokkos::pair(ng[2], lim2+ng[2]), 
                               buffer.edgeBuffer[2][2].d_view, ivar);

            //edge[2][3] : corner(1,1,0)<->corner(1,1,1)
            memcopy_VtoB(view, Kokkos::pair(lim0, lim0+ng[0]), 
                               Kokkos::pair(lim1, lim1+ng[1]),
                               Kokkos::pair(ng[2], lim2+ng[2]), 
                               buffer.edgeBuffer[2][3].d_view, ivar);
        }
    }   

    void copyToBuffer_faces(KV_cdouble_3d const view, Buffer & buffer, int const ivar)   
    {
        std::array<int, 3> ng = buffer.Nghost;
        int lim0 = view.extent(0)-2*ng[0];
        int lim1 = view.extent(1)-2*ng[1];
        int lim2 = view.extent(2)-2*ng[2];  

        for(int i=0; i<3; i++)
        {
            if(ng[i])
            {
                int Xstart = ng[0], Ystart=ng[1], Zstart=ng[2];
                int Xend = lim0+ng[0], Yend=lim1+ng[1], Zend=lim2+ng[2];
                for(int f=0; f<2; f++)
                {          
                    if(i==0) { Xstart = (f==0? ng[0]:lim0); Xend = (f==0?2*ng[0]:lim0+ng[0]); }
                    if(i==1) { Ystart = (f==0? ng[1]:lim1); Yend = (f==0?2*ng[1]:lim1+ng[1]); }
                    if(i==2) { Zstart = (f==0? ng[2]:lim2); Zend = (f==0?2*ng[2]:lim2+ng[2]); }

                    memcopy_VtoB(view, Kokkos::pair(Xstart, Xend), 
                                       Kokkos::pair(Ystart, Yend),
                                       Kokkos::pair(Zstart, Zend), 
                                       buffer.faceBuffer[i][f].d_view, ivar);
                }
            }
        }
    }

    void copyToBuffer(KV_cdouble_3d const view, Buffer & buffer, int const ivar) 
    {
        copyToBuffer_faces(view, buffer, ivar);  
        modifyDtoH(buffer.faceBuffer);  

        copyToBuffer_edges(view, buffer, ivar);
        modifyDtoH(buffer.edgeBuffer);

        copyToBuffer_corners(view, buffer, ivar);
        modifyDtoH(buffer.cornerBuffer);    

        Kokkos::fence();
    }   

    void copyFromBuffer_corners(KV_double_3d const view, Buffer const & buffer, int const ivar)   
    {
        std::array<int, 3> ng = buffer.Nghost;
        int lim0 = view.extent(0)-2*ng[0];
        int lim1 = view.extent(1)-2*ng[1];
        int lim2 = view.extent(2)-2*ng[2];  

        if(ng[0] || ng[1] || ng[2]) { return; }   

        //corner(0,0,0)
        memcopy_BtoV(view, Kokkos::pair(0, ng[0]), 
                           Kokkos::pair(0, ng[1]),
                           Kokkos::pair(0, ng[2]), 
                           buffer.cornerBuffer[0][0].d_view, ivar);

        //corner(1,0,0)  
        memcopy_BtoV(view, Kokkos::pair(lim0+ng[0], lim0+2*ng[0]), 
                           Kokkos::pair(0, ng[1]),
                           Kokkos::pair(0, ng[2]), 
                           buffer.cornerBuffer[0][1].d_view, ivar);

        //corner(0,1,0)  
        memcopy_BtoV(view, Kokkos::pair(0, ng[0]), 
                           Kokkos::pair(lim1+ng[1], lim1+2*ng[1]),
                           Kokkos::pair(0, ng[2]), 
                           buffer.cornerBuffer[0][2].d_view, ivar);   

        //corner(1,1,0)  
        memcopy_BtoV(view, Kokkos::pair(lim0+ng[0], lim0+2*ng[0]), 
                           Kokkos::pair(lim1+ng[1], lim1+2*ng[1]),
                           Kokkos::pair(0, ng[2]), 
                           buffer.cornerBuffer[0][3].d_view, ivar);   

        //corner(0,0,1) 
        memcopy_BtoV(view, Kokkos::pair(0, ng[0]), 
                           Kokkos::pair(0, ng[1]),
                           Kokkos::pair(lim2+ng[2], lim2+2*ng[2]), 
                           buffer.cornerBuffer[1][0].d_view, ivar);

        //corner(1,0,1)   
        memcopy_BtoV(view, Kokkos::pair(lim0+ng[0], lim0+2*ng[0]), 
                           Kokkos::pair(0, ng[1]),
                           Kokkos::pair(lim2+ng[2], lim2+2*ng[2]), 
                           buffer.cornerBuffer[1][1].d_view, ivar);

        //corner(0,1,1)
        memcopy_BtoV(view, Kokkos::pair(0, ng[0]), 
                           Kokkos::pair(lim1+ng[1], lim1+2*ng[1]),
                           Kokkos::pair(lim2+ng[2], lim2+2*ng[2]),
                           buffer.cornerBuffer[1][2].d_view, ivar);

        //corner(1,1,1)
        memcopy_BtoV(view, Kokkos::pair(lim0+ng[0], lim0+2*ng[0]), 
                           Kokkos::pair(lim1+ng[1], lim1+2*ng[1]),
                           Kokkos::pair(lim2+ng[2], lim2+2*ng[2]), 
                           buffer.cornerBuffer[1][3].d_view, ivar);
    }   

    void copyFromBuffer_edges(KV_double_3d const view, Buffer const & buffer, int const ivar) 
    {
        std::array<int, 3> ng = buffer.Nghost;
        int lim0 = view.extent(0)-2*ng[0];
        int lim1 = view.extent(1)-2*ng[1];
        int lim2 = view.extent(2)-2*ng[2];  

        if(ng[1] && ng[2])
        {
            //edge[0][0] : corner(0,0,0)<->corner(1,0,0)
            memcopy_BtoV(view, Kokkos::pair(ng[0], lim0+ng[0]), 
                               Kokkos::pair(0, ng[1]),
                               Kokkos::pair(0, ng[2]), 
                               buffer.edgeBuffer[0][0].d_view, ivar);

            //edge[0][1] : corner(0,1,0)<->corner(1,1,0)
            memcopy_BtoV(view, Kokkos::pair(ng[0], lim0+ng[0]), 
                               Kokkos::pair(lim1+ng[1], lim1+2*ng[1]),
                               Kokkos::pair(0, ng[2]), 
                               buffer.edgeBuffer[0][1].d_view, ivar);    

            //edge[0][2] : corner(0,0,1)<->corner(1,0,1)
            memcopy_BtoV(view, Kokkos::pair(ng[0], lim0+ng[0]), 
                               Kokkos::pair(0, ng[1]),
                               Kokkos::pair(lim2+ng[2], lim2+2*ng[2]), 
                               buffer.edgeBuffer[0][2].d_view, ivar);

            //edge[0][3] : corner(0,1,1)<->corner(1,1,1)
            memcopy_BtoV(view, Kokkos::pair(ng[0], lim0+ng[0]), 
                               Kokkos::pair(lim1+ng[1], lim1+2*ng[1]),
                               Kokkos::pair(lim2+ng[2], lim2+2*ng[2]), 
                               buffer.edgeBuffer[0][3].d_view, ivar);
        }   

        if(ng[0] && ng[2])
        {
            //edge[1][0] : corner(0,0,0)<->corner(0,1,0)    
            memcopy_BtoV(view, Kokkos::pair(0, ng[0]), 
                               Kokkos::pair(ng[1], lim1+ng[1]),
                               Kokkos::pair(0, ng[2]), 
                               buffer.edgeBuffer[1][0].d_view, ivar);

            //edge[1][1] : corner(1,0,0)<->corner(1,1,0)
            memcopy_BtoV(view, Kokkos::pair(lim0+ng[0], lim0+2*ng[0]),
                               Kokkos::pair(ng[1], lim1+ng[1]),
                               Kokkos::pair(0, ng[2]),
                               buffer.edgeBuffer[1][1].d_view, ivar);

            //edge[1][2] : corner(0,0,1)<->corner(0,1,1)
            memcopy_BtoV(view, Kokkos::pair(0, ng[0]),
                               Kokkos::pair(ng[1], lim1+ng[1]),
                               Kokkos::pair(lim2+ng[2], lim2+2*ng[2]),
                               buffer.edgeBuffer[1][2].d_view, ivar);

            //edge[1][3] : corner(1,0,1)<->corner(1,1,1)
            memcopy_BtoV(view, Kokkos::pair(lim0+ng[0], lim0+2*ng[0]),
                               Kokkos::pair(ng[1], lim1+ng[1]),
                               Kokkos::pair(lim2+ng[2], lim2+2*ng[2]),
                               buffer.edgeBuffer[1][3].d_view, ivar);
        }

        if(ng[0] && ng[1])
        {
            //edge[2][0] : corner(0,0,0)<->corner(0,0,1)  
            memcopy_BtoV(view, Kokkos::pair(0, ng[0]),
                               Kokkos::pair(0, ng[1]),
                               Kokkos::pair(ng[2], lim2+ng[2]),
                               buffer.edgeBuffer[2][0].d_view, ivar);

            //edge[2][1] : corner(1,0,0)<->corner(1,0,1)
            memcopy_BtoV(view, Kokkos::pair(lim0+ng[0], lim0+2*ng[0]),
                               Kokkos::pair(0, ng[1]),
                               Kokkos::pair(ng[2], lim2+ng[2]),
                               buffer.edgeBuffer[2][1].d_view, ivar);

            //edge[2][2] : corner(0,1,0)<->corner(0,1,1)
            memcopy_BtoV(view, Kokkos::pair(0, ng[0]),
                               Kokkos::pair(lim1+ng[1], lim1+2*ng[1]),
                               Kokkos::pair(ng[2], lim2+ng[2]),
                               buffer.edgeBuffer[2][2].d_view, ivar);

            //edge[2][3] : corner(1,1,0)<->corner(1,1,1)
            memcopy_BtoV(view, Kokkos::pair(lim0+ng[0], lim0+2*ng[0]),
                               Kokkos::pair(lim1+ng[1], lim1+2*ng[1]),
                               Kokkos::pair(ng[2], lim2+ng[2]),
                               buffer.edgeBuffer[2][3].d_view, ivar);
        }
    }

    void copyFromBuffer_faces(KV_double_3d const view, Buffer const & buffer, int const ivar) 
    {
        std::array<int, 3> ng = buffer.Nghost;
        int lim0 = view.extent(0)-2*ng[0];
        int lim1 = view.extent(1)-2*ng[1];
        int lim2 = view.extent(2)-2*ng[2];

        for(int i=0; i<3; i++)
        {
            if(ng[i])
            {
                int Xstart = ng[0], Ystart=ng[1], Zstart=ng[2];
                int Xend = lim0+ng[0], Yend=lim1+ng[1], Zend=lim2+ng[2];
                for(int f=0; f<2; f++)
                {          
                    if(i==0) { Xstart = f*(lim0+ng[0]); Xend = f*(lim0+ng[0])+ng[0]; }
                    if(i==1) { Ystart = f*(lim1+ng[1]); Yend = f*(lim1+ng[1])+ng[1]; }
                    if(i==2) { Zstart = f*(lim2+ng[2]); Zend = f*(lim2+ng[2])+ng[2]; }

                    memcopy_BtoV(view, Kokkos::pair(Xstart, Xend),
                                       Kokkos::pair(Ystart, Yend),
                                       Kokkos::pair(Zstart, Zend),
                                       buffer.faceBuffer[i][f].d_view, ivar);
                }
            }
        }
    }

    void copyFromBuffer(KV_double_3d const view, Buffer & buffer, int const ivar)   
    {
        modifyHtoD(buffer.faceBuffer);
        copyFromBuffer_faces(view, buffer, ivar);  

        modifyHtoD(buffer.faceBuffer);
        copyFromBuffer_edges(view, buffer, ivar);

        modifyHtoD(buffer.faceBuffer);
        copyFromBuffer_corners(view, buffer, ivar); 

        Kokkos::fence();
    }

    void exchangeBuffer_corners(Buffer & send_buffer, Buffer & recv_buffer, std::array<int, 3> const & ng, MPI_Comm const & comm_cart, std::array<std::array<std::array<int, 3>, 3>, 3> const & nrank) 
    {    
        if(ng[0] || ng[1] || ng[2]) { return; }   

        int dst, src, count;
        count = send_buffer.cornerBuffer[0][0].h_view.size();   

        //corner[0][0] <=> corner[1][3]
        dst = nrank[0][0][0];
        src = nrank[2][2][2];
        MPI_Sendrecv(send_buffer.cornerBuffer[0][0].h_view.data(), count, MPI_DOUBLE, dst, 0,
                     recv_buffer.cornerBuffer[1][3].h_view.data(), count, MPI_DOUBLE, src, 0,
                     comm_cart, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_buffer.cornerBuffer[1][3].h_view.data(), count, MPI_DOUBLE, src, 0,
                     recv_buffer.cornerBuffer[0][0].h_view.data(), count, MPI_DOUBLE, dst, 0,
                     comm_cart, MPI_STATUS_IGNORE);

        //corner[0][1] <=> corner[1][2]
        dst = nrank[2][0][0];
        src = nrank[0][2][2];
        MPI_Sendrecv(send_buffer.cornerBuffer[0][1].h_view.data(), count, MPI_DOUBLE, dst, 0,
                     recv_buffer.cornerBuffer[1][2].h_view.data(), count, MPI_DOUBLE, src, 0,
                     comm_cart, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_buffer.cornerBuffer[1][2].h_view.data(), count, MPI_DOUBLE, src, 0,
                     recv_buffer.cornerBuffer[0][1].h_view.data(), count, MPI_DOUBLE, dst, 0,
                     comm_cart, MPI_STATUS_IGNORE);

        //corner[0][2] <=> corner[1][1]
        dst = nrank[0][2][0];
        src = nrank[2][0][2];
        MPI_Sendrecv(send_buffer.cornerBuffer[0][2].h_view.data(), count, MPI_DOUBLE, dst, 0,
                     recv_buffer.cornerBuffer[1][1].h_view.data(), count, MPI_DOUBLE, src, 0,
                     comm_cart, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_buffer.cornerBuffer[1][1].h_view.data(), count, MPI_DOUBLE, src, 0,
                     recv_buffer.cornerBuffer[0][2].h_view.data(), count, MPI_DOUBLE, dst, 0,
                     comm_cart, MPI_STATUS_IGNORE); 

        //corner[0][3] <=> corner[1][0]
        dst = nrank[2][2][0];
        src = nrank[0][0][2];
        MPI_Sendrecv(send_buffer.cornerBuffer[0][3].h_view.data(), count, MPI_DOUBLE, dst, 0,
                     recv_buffer.cornerBuffer[1][0].h_view.data(), count, MPI_DOUBLE, src, 0,
                     comm_cart, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_buffer.cornerBuffer[1][0].h_view.data(), count, MPI_DOUBLE, src, 0,
                     recv_buffer.cornerBuffer[0][3].h_view.data(), count, MPI_DOUBLE, dst, 0,
                     comm_cart, MPI_STATUS_IGNORE); 

    }   

    void exchangeBuffer_edges(Buffer & send_buffer, Buffer & recv_buffer, std::array<int, 3> const & ng, MPI_Comm const & comm_cart, std::array<std::array<std::array<int, 3>, 3>, 3> const & nrank)   
    {    
        int dst, src, count;

        if(ng[1] && ng[2])
        {
            count = send_buffer.edgeBuffer[0][0].h_view.size();
            dst = nrank[1][0][0];
            src = nrank[1][2][2];
            MPI_Sendrecv(send_buffer.edgeBuffer[0][0].h_view.data(), count, MPI_DOUBLE, dst, 0,
                         recv_buffer.edgeBuffer[0][3].h_view.data(), count, MPI_DOUBLE, src, 0,
                         comm_cart, MPI_STATUS_IGNORE);
            MPI_Sendrecv(send_buffer.edgeBuffer[0][3].h_view.data(), count, MPI_DOUBLE, src, 0,
                         recv_buffer.edgeBuffer[0][0].h_view.data(), count, MPI_DOUBLE, dst, 0,
                         comm_cart, MPI_STATUS_IGNORE);

            dst = nrank[1][2][0];
            src = nrank[1][0][2];
            MPI_Sendrecv(send_buffer.edgeBuffer[0][1].h_view.data(), count, MPI_DOUBLE, dst, 0,
                         recv_buffer.edgeBuffer[0][2].h_view.data(), count, MPI_DOUBLE, src, 0,
                         comm_cart, MPI_STATUS_IGNORE);
            MPI_Sendrecv(send_buffer.edgeBuffer[0][2].h_view.data(), count, MPI_DOUBLE, src, 0,
                         recv_buffer.edgeBuffer[0][1].h_view.data(), count, MPI_DOUBLE, dst, 0,
                         comm_cart, MPI_STATUS_IGNORE);
        }

        if(ng[0] && ng[2])
        {
            count = send_buffer.edgeBuffer[1][0].h_view.size();
            dst = nrank[0][1][0];
            src = nrank[2][1][2];
            MPI_Sendrecv(send_buffer.edgeBuffer[1][0].h_view.data(), count, MPI_DOUBLE, dst, 1,
                         recv_buffer.edgeBuffer[1][3].h_view.data(), count, MPI_DOUBLE, src, 1,
                         comm_cart, MPI_STATUS_IGNORE);
            MPI_Sendrecv(send_buffer.edgeBuffer[1][3].h_view.data(), count, MPI_DOUBLE, src, 1,
                         recv_buffer.edgeBuffer[1][0].h_view.data(), count, MPI_DOUBLE, dst, 1,
                         comm_cart, MPI_STATUS_IGNORE);

            dst = nrank[0][1][2];
            src = nrank[2][1][0];
            MPI_Sendrecv(send_buffer.edgeBuffer[1][2].h_view.data(), count, MPI_DOUBLE, dst, 1,
                         recv_buffer.edgeBuffer[1][1].h_view.data(), count, MPI_DOUBLE, src, 1,
                         comm_cart, MPI_STATUS_IGNORE);
            MPI_Sendrecv(send_buffer.edgeBuffer[1][1].h_view.data(), count, MPI_DOUBLE, src, 1,
                         recv_buffer.edgeBuffer[1][2].h_view.data(), count, MPI_DOUBLE, dst, 1,
                         comm_cart, MPI_STATUS_IGNORE);
        }

        if(ng[0] && ng[1])
        {
            count = send_buffer.edgeBuffer[2][0].h_view.size();
            dst = nrank[0][0][1];
            src = nrank[2][2][1];
            MPI_Sendrecv(send_buffer.edgeBuffer[2][0].h_view.data(), count, MPI_DOUBLE, dst, 2,
                         recv_buffer.edgeBuffer[2][3].h_view.data(), count, MPI_DOUBLE, src, 2,
                         comm_cart, MPI_STATUS_IGNORE);
            MPI_Sendrecv(send_buffer.edgeBuffer[2][3].h_view.data(), count, MPI_DOUBLE, src, 2,
                         recv_buffer.edgeBuffer[2][0].h_view.data(), count, MPI_DOUBLE, dst, 2,
                         comm_cart, MPI_STATUS_IGNORE);
            dst = nrank[2][0][1];
            src = nrank[0][2][1];
            MPI_Sendrecv(send_buffer.edgeBuffer[2][1].h_view.data(), count, MPI_DOUBLE, dst, 0,
                         recv_buffer.edgeBuffer[2][2].h_view.data(), count, MPI_DOUBLE, src, 0,
                         comm_cart, MPI_STATUS_IGNORE);
            MPI_Sendrecv(send_buffer.edgeBuffer[2][2].h_view.data(), count, MPI_DOUBLE, src, 0,
                         recv_buffer.edgeBuffer[2][1].h_view.data(), count, MPI_DOUBLE, dst, 0,
                         comm_cart, MPI_STATUS_IGNORE);
        }  
    }

    void exchangeBuffer_faces(Buffer & send_buffer, Buffer & recv_buffer, std::array<int, 3> const & ng, MPI_Comm const & comm_cart)   
    {    
        int dst, src, count;

        for(int i=0; i<3; i++)
        {
            if(ng[i])
            {
                MPI_Cart_shift(comm_cart, i, -1, &src, &dst);
                count = send_buffer.faceBuffer[i][0].h_view.size();
                //send to left, recv from right
                MPI_Sendrecv(send_buffer.faceBuffer[i][0].h_view.data(), count, MPI_DOUBLE, dst, i,
                             recv_buffer.faceBuffer[i][1].h_view.data(), count, MPI_DOUBLE, src, i,
                             comm_cart, MPI_STATUS_IGNORE);

                //send to right, recv from left
                MPI_Sendrecv(send_buffer.faceBuffer[i][1].h_view.data(), count, MPI_DOUBLE, src, i,
                             recv_buffer.faceBuffer[i][0].h_view.data(), count, MPI_DOUBLE, dst, i,
                             comm_cart, MPI_STATUS_IGNORE);
            }
        }
    }

    void exchangeBuffer(Buffer & send_buffer, Buffer & recv_buffer, std::array<int,3> const & ng, MPI_Comm const & comm_cart, std::array<std::array<std::array<int, 3>, 3>, 3> const & nrank) 
    {
        exchangeBuffer_faces(send_buffer, recv_buffer, ng, comm_cart);
        exchangeBuffer_edges(send_buffer, recv_buffer, ng, comm_cart, nrank);
        exchangeBuffer_corners(send_buffer, recv_buffer, ng, comm_cart, nrank);
        Kokkos::fence();
    }
    

    template<std::size_t n, std::size_t m>
    void modifyDtoH(std::array<std::array<KDV_double_4d, n >, m > & buffer_element)
    {
        for(size_t i=0; i<m; i++)
        {
            for(size_t j=0; j<n; j++)
            {
                buffer_element[i][j].modify_device() ;
                buffer_element[i][j].sync_host() ;
            }
        }
    }   

    template<std::size_t n, std::size_t m>
    void modifyHtoD(std::array<std::array<KDV_double_4d,n>,m> & buffer_element)
    {
        for(size_t i=0; i<m; i++)
        {
            for(size_t j=0; j<n; j++)
            {
                buffer_element[i][j].modify_host();
                buffer_element[i][j].sync_device();
            }
        }
    }
} // end of anonymous namespace

} // namespace novapp

namespace novapp
{


void IBoundaryCondition::ghostExchange(KV_double_3d rho,
                                       KV_double_4d rhou,
                                       KV_double_3d E, 
                                       Grid const & grid)
{
    copyToBuffer(rho,  sbuf, 0);
    copyToBuffer(E,    sbuf, 1); 
    
    for(int n=0; n<ndim; n++)
    {   
        copyToBuffer(Kokkos::subview(rhou, ALL, ALL, ALL, n), sbuf, 2+n);
    }

    exchangeBuffer(sbuf, rbuf, grid.Nghost, grid.comm_cart, grid.NeighborRank);

    copyFromBuffer(rho,  rbuf, 0);
    copyFromBuffer(E,    rbuf, 1);
    for(int n=0; n<ndim; n++)
    {   
        copyFromBuffer(Kokkos::subview(rhou, ALL, ALL, ALL, n), rbuf, 2+n);
    }
}

} // namespace novapp

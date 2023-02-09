/**
 * @file range.cpp
 * Geom class implementation
 */

#include "range.hpp"

void Range::Compute_range(std::array<int,3> Cmin, std::array<int,3> Cmax, std::array<int,3> Nghost ) 
{
    for (int idim=0 ;idim<3 ; idim++){
        Corner_min[idim] = Cmin[idim] ;
        Corner_max[idim] = Cmax[idim] ;

        int NN = Corner_max[idim]-Corner_min[idim] ;

        Nc_min_2g[idim] = 0 ;
        Nc_max_2g[idim] = NN + 2*Nghost[idim] ;
        Nf_min_2g[idim] = 0 ;
        Nf_max_2g[idim] = NN + 2*Nghost[idim] + 1 ;

        Nc_min_1g[idim] = 1 ;
        Nc_max_1g[idim] = NN + Nghost[idim]; 
        Nf_min_1g[idim] = 1 ;
        Nf_max_1g[idim] = NN + Nghost[idim]+ 1 ; 

        Nc_min_0g[idim] = 2  ;
        Nc_max_0g[idim] = NN ; 
        Nf_min_0g[idim] = 2  ;
        Nf_max_0g[idim] = NN + 1 ; 
    }
}


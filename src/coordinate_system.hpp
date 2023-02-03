//!
//! @file coordinate_system.hpp
// Choice for the coordinate system
//!

#pragma once
#include <map>
#include <exception>

enum system_choice_enum{
    Cartesian   = 0,
    Cylindrical = 1,
    Spherical   = 2,
};


//! Conversion primary to conservative variables
//! @param[in] s string system name
//! @param[out] x alpha value
system_choice_enum GetenumIndex( std::string s ) {    
    static std::map<std::string,system_choice_enum> string2choice {
       { "Cartesian",   Cartesian }, 
       { "Cylindrical", Cylindrical },
       { "Spherical",   Spherical }  
    };
    auto x = string2choice.find(s);
    if(x != std::end(string2choice)) {
        return x->second;
    }
    throw std::invalid_argument("s");
}

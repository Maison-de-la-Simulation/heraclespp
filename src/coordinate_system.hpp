//!
//! @file coordinate_system.hpp
//!

#pragma once
#include <map>
#include <exception>

enum system_choice_enum{
    Cartesian   = 0,
    Cylindrical = 1,
    Spherical   = 2,
};


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
};
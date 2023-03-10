//!
//! @file coordinate_system.hpp
// Choice for the coordinate system
//!

#pragma once

#include <map>
#include <string>

namespace novapp
{

enum system_choice_enum
{
    Cartesian = 0,
    Cylindrical = 1,
    Spherical = 2,
};

inline system_choice_enum GetenumIndex(std::string const& label)
{
    static const std::map<std::string, system_choice_enum> string2choice {
            {"Cartesian", Cartesian},
            {"Cylindrical", Cylindrical},
            {"Spherical", Spherical}};
    return string2choice.at(label);
}

} // namespace novapp

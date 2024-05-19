#pragma once

#include <iomanip>
#include <ios>
#include <ostream>
#include <string_view>

namespace novapp {

template <class T>
void print_info(std::ostream& os, std::string_view const var_name, T const var_value)
{
    using namespace std;
    os << left << setw(40) << setfill('.') << var_name;
    os << right << setw(40) << setfill('.') << var_value << '\n';
}

} // namespace novapp

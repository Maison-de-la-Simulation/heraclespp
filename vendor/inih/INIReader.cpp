// Read an INI file into easy-to-access name/value pairs.

// inih and INIReader are released under the New BSD license (see LICENSE.txt).
// Go to the project home page for more info:
//
// https://github.com/benhoyt/inih

#include "INIReader.hpp"
#include <algorithm>  // for transform
#include <cctype>     // for tolower
#include <stdexcept>
#include <limits>
#include <string>
#include <utility>    // for pair
#include "ini.h"      // for ini_parse

INIReader::INIReader() = default;

INIReader::INIReader(const std::string& filename)
{
    int const ec = ini_parse(filename.c_str(), ValueHandler, this);
    if (ec != 0)
    {
        if (ec == -1)
        {
            throw std::runtime_error("Could not open the file");
        }
        throw std::runtime_error(std::string("Parsing error at line ") + std::to_string(ec));
    }
}

std::string INIReader::Get(const std::string& section, const std::string& name, const std::string& default_value) const
{
    std::string key = MakeKey(section, name);
    // Use _values.find() here instead of _values.at() to support pre C++11 compilers
    return _values.count(key) ? _values.find(key)->second : default_value;
}

unsigned INIReader::GetInteger(const std::string& section, const std::string& name, unsigned default_value) const
{
    const std::string valstr = Get(section, name, "");
    unsigned long result = valstr.empty() ? default_value : std::stoul(valstr);
    if (result > std::numeric_limits<unsigned>::max())
    {
        throw std::out_of_range("stou");
    }
    return valstr.empty() ? default_value : static_cast<unsigned>(result);
}

unsigned long INIReader::GetInteger(const std::string& section, const std::string& name, unsigned long default_value) const
{
    const std::string valstr = Get(section, name, "");
    // This parses "1234" (decimal) and also "0x4D2" (hex)
    return valstr.empty() ? default_value : std::stoul(valstr);
}

unsigned long long INIReader::GetInteger(const std::string& section, const std::string& name, unsigned long long default_value) const
{
    const std::string valstr = Get(section, name, "");
    // This parses "1234" (decimal) and also "0x4D2" (hex)
    return valstr.empty() ? default_value : std::stoull(valstr);
}

int INIReader::GetInteger(const std::string& section, const std::string& name, int default_value) const
{
    const std::string valstr = Get(section, name, "");
    // This parses "1234" (decimal) and also "0x4D2" (hex)
    return valstr.empty() ? default_value : std::stoi(valstr);
}

long INIReader::GetInteger(const std::string& section, const std::string& name, long default_value) const
{
    const std::string valstr = Get(section, name, "");
    // This parses "1234" (decimal) and also "0x4D2" (hex)
    return valstr.empty() ? default_value : std::stol(valstr);
}

long long INIReader::GetInteger(const std::string& section, const std::string& name, long long default_value) const
{
    const std::string valstr = Get(section, name, "");
    // This parses "1234" (decimal) and also "0x4D2" (hex)
    return valstr.empty() ? default_value : std::stoll(valstr);
}

float INIReader::GetReal(const std::string& section, const std::string& name, float default_value) const
{
    const std::string valstr = Get(section, name, "");
    return valstr.empty() ?  default_value : std::stof(valstr);
}

double INIReader::GetReal(const std::string& section, const std::string& name, double default_value) const
{
    const std::string valstr = Get(section, name, "");
    return valstr.empty() ?  default_value : std::stod(valstr);
}

long double INIReader::GetReal(const std::string& section, const std::string& name, long double default_value) const
{
    const std::string valstr = Get(section, name, "");
    return valstr.empty() ?  default_value : std::stold(valstr);
}

bool INIReader::GetBoolean(const std::string& section, const std::string& name, bool default_value) const
{
    const auto stob = [](std::string const& str) -> bool
    {
        std::string lowered_str = str;
        // Convert to lower case to make std::string comparisons case-insensitive
        std::transform(str.begin(), str.end(), lowered_str.begin(), ::tolower);
        if (lowered_str == "true" || lowered_str == "yes" || lowered_str == "on" || lowered_str == "1")
            return true;
        if (lowered_str == "false" || lowered_str == "no" || lowered_str == "off" || lowered_str == "0")
            return false;
        throw std::out_of_range("stob");
    };
    const std::string valstr = Get(section, name, "");
    return valstr.empty() ? default_value : stob(valstr);
}

std::string INIReader::MakeKey(const std::string& section, const std::string& name)
{
    std::string key = section + "=" + name;
    // Convert to lower case to make section/name lookups case-insensitive
    std::transform(key.begin(), key.end(), key.begin(), ::tolower);
    return key;
}

int INIReader::ValueHandler(void* user, const char* section, const char* name,
                            const char* value)
{
    INIReader* reader = (INIReader*)user;
    std::string key = MakeKey(section, name);
    if (reader->_values[key].size() > 0)
        reader->_values[key] += "\n";
    reader->_values[key] += value;
    return 1;
}

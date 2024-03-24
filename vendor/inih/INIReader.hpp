// Read an INI file into easy-to-access name/value pairs.

// inih and INIReader are released under the New BSD license (see LICENSE.txt).
// Go to the project home page for more info:
//
// https://github.com/benhoyt/inih

#ifndef __INIREADER_H__
#define __INIREADER_H__

#include <map>
#include <string>

// Read an INI file into easy-to-access name/value pairs. (Note that I've gone
// for simplicity here rather than speed, but it should be pretty decent.)
class INIReader
{
public:
    // Construct INIReader and parse given filename. See ini.h for more info
    // about the parsing.
    INIReader(const std::string& filename);

    // Get a string value from INI file, returning default_value if not found.
    std::string Get(const std::string& section, const std::string& name,
                    const std::string& default_value) const;

    // Get an integer (unsigned) value from INI file, returning default_value if
    // not found or not a valid integer (decimal "1234", "-1234", or hex "0x4d2").
    unsigned GetInteger(const std::string& section, const std::string& name, unsigned default_value) const;

    // Get an integer (unsigned long) value from INI file, returning default_value if
    // not found or not a valid integer (decimal "1234", "-1234", or hex "0x4d2").
    unsigned long GetInteger(const std::string& section, const std::string& name, unsigned long default_value) const;

    // Get an integer (unsigned long long) value from INI file, returning default_value if
    // not found or not a valid integer (decimal "1234", "-1234", or hex "0x4d2").
    unsigned long long GetInteger(const std::string& section, const std::string& name, unsigned long long default_value) const;

    // Get an integer (int) value from INI file, returning default_value if
    // not found or not a valid integer (decimal "1234", "-1234", or hex "0x4d2").
    int GetInteger(const std::string& section, const std::string& name, int default_value) const;

    // Get an integer (long) value from INI file, returning default_value if
    // not found or not a valid integer (decimal "1234", "-1234", or hex "0x4d2").
    long GetInteger(const std::string& section, const std::string& name, long default_value) const;

    // Get an integer (long long) value from INI file, returning default_value if
    // not found or not a valid integer (decimal "1234", "-1234", or hex "0x4d2").
    long long GetInteger(const std::string& section, const std::string& name, long long default_value) const;

    // Get a real (floating point float) value from INI file, returning
    // default_value if not found or not a valid floating point value
    // according to stof().
    float GetReal(const std::string& section, const std::string& name, float default_value) const;

    // Get a real (floating point double) value from INI file, returning
    // default_value if not found or not a valid floating point value
    // according to stod().
    double GetReal(const std::string& section, const std::string& name, double default_value) const;

    // Get a real (floating point long double) value from INI file, returning
    // default_value if not found or not a valid floating point value
    // according to stold().
    long double GetReal(const std::string& section, const std::string& name, long double default_value) const;

    // Get a boolean value from INI file, returning default_value if not found or if
    // not a valid true/false value. Valid true values are "true", "yes", "on", "1",
    // and valid false values are "false", "no", "off", "0" (not case sensitive).
    bool GetBoolean(const std::string& section, const std::string& name, bool default_value) const;

private:
    std::map<std::string, std::string> _values;
    static std::string MakeKey(const std::string& section, const std::string& name);
    static int ValueHandler(void* user, const char* section, const char* name,
                            const char* value);
};

#endif  // __INIREADER_H__

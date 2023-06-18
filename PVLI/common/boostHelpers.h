#pragma once

#include <string>
#include <iostream>
#include <boost/preprocessor.hpp>

//Found here: https://stackoverflow.com/questions/5093460/how-to-convert-an-enum-type-variable-to-a-string

template<typename T>
inline T from_string(const std::string& ){ return T(); }

#define X_ENUM_WITH_STRINGS_CONVERSIONS_TOSTRING_CASE(r, name, elem)          \
    case name::elem : return BOOST_PP_STRINGIZE(elem);

#define X_ENUM_WITH_STRINGS_CONVERSIONS_FROMSTRING_CASE(r, name, elem)        \
    if(s == BOOST_PP_STRINGIZE(elem)) return name::elem;

#define ENUM_STRINGS(name, unknown, enumerators)                              \
                                                                              \
    inline std::string to_string(const name& v) {                             \
        switch (v) {                                                          \
            BOOST_PP_SEQ_FOR_EACH(                                            \
                X_ENUM_WITH_STRINGS_CONVERSIONS_TOSTRING_CASE,                \
                name, enumerators                                             \
            )                                                                 \
            default: return "[Unknown " BOOST_PP_STRINGIZE(name) "]";         \
        }                                                                     \
    }                                                                         \
                                                                              \
    template<> inline name from_string<name>(const std::string& s) {          \
        BOOST_PP_SEQ_FOR_EACH(                                                \
            X_ENUM_WITH_STRINGS_CONVERSIONS_FROMSTRING_CASE,                  \
            name, enumerators                                                 \
        )                                                                     \
        return name::unknown;                                                 \
    }

#define ENUM_STREAM_OPERATORS(name)                                           \
namespace std {                                                               \
    inline ostream &operator<<(ostream &out, const ::name& v) {               \
        out << ::to_string(v);                                                \
        return out;                                                           \
    }                                                                         \
    inline istream &operator>>(istream &in, ::name& v) {                      \
        string s;                                                             \
        in >> s;                                                              \
        v = ::from_string<::name>(s);                                         \
        return in;                                                            \
    }                                                                         \
}

#define ENUM_WITH_STRINGS(name, type, unknown, enumerators)                   \
    enum class name : type {                                                  \
        BOOST_PP_SEQ_ENUM(enumerators)                                        \
    };                                                                        \
    ENUM_STRINGS(name, unknown, enumerators)

#define ENUM_STRINGS_STREAM_OPS(name, type, unknown, enumerators)             \
    ENUM_WITH_STRINGS(name, type, unknown, enumerators)                       \
    ENUM_STREAM_OPERATORS(name)

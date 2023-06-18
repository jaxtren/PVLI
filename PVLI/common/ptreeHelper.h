#pragma once

#include <string>
#include <iostream>
#include <sstream>

#define BOOST_BIND_GLOBAL_PLACEHOLDERS
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>
#include <boost/property_tree/json_parser.hpp>

namespace pt = boost::property_tree;
namespace boost { namespace property_tree {

    void write_value(std::ostream&, const ptree &, int precision = 3);
    void write_spaces(std::ostream&, const ptree&, int offset = 0, int spaces = 2, int precision = 3);
    void merge(ptree &dst, const ptree &src, const ptree::path_type& path = "");
    bool read_info_ext(const std::string& file, ptree& tree);

    inline std::string to_string_value(const ptree& tree, int precision = 3) {
        std::stringstream ss;
        write_value(ss, tree, precision);
        return ss.str();
    }

    inline std::string to_string_spaces(const ptree& tree, int offset = 0, int spaces = 2, int precision = 3) {
        std::stringstream ss;
        write_spaces(ss, tree, offset, spaces, precision);
        return ss.str();
    }

    inline std::string to_string_info(const ptree& tree) {
        try {
            std::stringstream ss;
            pt::write_info(ss, tree);
            return ss.str();
        }catch (const pt::info_parser_error&){
            return "";
        }
    }

    inline bool read_info_string(const std::string& s, ptree& tree) {
        try {
            std::stringstream ss(s);
            pt::read_info(ss, tree);
        } catch (const pt::info_parser_error&) {
            return false;
        }
        return true;
    }
} }
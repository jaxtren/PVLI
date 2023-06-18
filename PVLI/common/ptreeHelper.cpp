#include "ptreeHelper.h"
#include <sstream>
#include <iomanip>
#include "glmHelpers.h"

#include <filesystem>
namespace fs = std::filesystem;

using namespace std;

namespace boost { namespace property_tree {

    void write_value(ostream &out, const ptree &tree, int precision) {
        auto i = tree.get_value_optional<int>();
        auto v = tree.get_value_optional<glm::vec3>();
        auto d = tree.get_value_optional<double>();
        if (v) out << fixed << setprecision(precision) << v->x << ' ' << v->y << ' ' << v->z;
        else if (d) {
            if (*d == floor(*d)) out << (long) (*d);
            else out << fixed << setprecision(precision) << *d;
        } else if (!tree.data().empty()) out << tree.data();
    }

    void write_spaces(ostream &out, const ptree &tree, int offset, int spaces, int precision) {
        string prefix(offset * spaces, ' ');
        for (auto &t : tree) {
            out << prefix << t.first;
            if (!t.second.data().empty()) {
                out << ' ';
                write_value(out, t.second, precision);
            }
            out << endl;
            write_spaces(out, t.second, offset + 1, spaces, precision);
        }
    }

    void merge(ptree &dst, const ptree &src, const ptree::path_type& path) {
        if(!src.data().empty() || !dst.get_child_optional(path))
            dst.put(path, src.data());
        for(auto& t : src)
            merge(dst, t.second, path / ptree::path_type(t.first));
    }

    bool read_info_ext(const std::string& file, ptree& tree) {
        try {
            pt::read_info(file, tree);
            auto base = tree.get_optional<std::string>("base");
            if(base){
                fs::path current = file, path = *base;
                if(path.is_relative())
                    path = current.parent_path() / path;

                if(current.lexically_normal() == path.lexically_normal())
                    return false;

                pt::ptree t;
                if(!read_info_ext(path.string(), t)) return false;
                swap(tree, t);
                pt::merge(tree, t);
            }

        } catch (const pt::info_parser_error&) {
            return false;
        }
        return true;
    }
} }
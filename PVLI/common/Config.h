#pragma once
#include "ptreeHelper.h"
#include <functional>
#include <map>
#include <set>
#include <string>

class Config {
public:
    pt::ptree* tree = nullptr;

    inline Config() = default;
    inline Config(pt::ptree& t) : tree(&t) {}

    inline Config get(const std::string& name) const {
        if (!tree) return Config();
        auto c = tree->get_child_optional(name);
        return c ? Config(*c) : Config();
    }

    inline Config operator[](const std::string& name) const { return get(name); }

    inline Config create(const std::string& name) const {
        if (!tree) return Config();
        auto c = tree->get_child_optional(name);
        return Config(c ? *c : tree->put_child(name, pt::ptree()));
    }

    /**
     * get property
     * @tparam T type
     * @param name property name
     * @param value
     * @return true when property was found and had different value
     */
    template<typename T>
    inline bool get(const std::string& name, T& value) const {
        if (!tree) return false;
        auto v = tree->get_optional<T>(name);
        if (v && *v != value) {
            value = *v;
            return true;
        }
        return false;
    }

    template<typename MapT = std::string, typename T>
    inline bool get(const std::string& name, T& value, const std::map<MapT, T> map) const {
        if (!tree) return false;
        auto v = tree->get_optional<MapT>(name);
        if (v) {
            auto it = map.find(*v);
            if (it != map.end() && it->second != value) {
                value = it->second;
                return true;
            }
        }
        return false;
    }

    /**
     * compare property
     * @tparam T type
     * @param name property name
     * @param value
     * @return true when property was found and had different value
     */
    template<typename T>
    inline bool diff(const std::string& name, const T& value) const {
        if (!tree) return false;
        auto v = tree->get_optional<T>(name);
        return v && *v != value;
    }

    template<typename T>
    inline void set(const std::string& name, const T& value) {
        if (tree) tree->put(name, value);
    }

    inline bool contains(const std::string& name) const {
        return tree && (bool) tree->get_child_optional(name);
    }

    template<typename ...Args>
    static auto anyChanged(Args ...args)
    {
        return (args || ...);
    }
};

class ConfigMapper {
    using function = std::function<bool(const std::string, Config)>;
    std::map<std::string, std::pair<function, function>> map;
    std::map<void*, std::string> changedMap;
    std::set<std::string> changedValues;

public:

    template<typename T>
    ConfigMapper& reg(const std::string& name, T& value) {
        map[name] = {
            [value = &value](auto n, auto cfg) { return cfg.get(n, *value); },
            [value = &value](auto n, auto cfg) { cfg.set(n, *value); return true; }
        };
        changedMap[(void*) &value] = name;
        return *this;
    }

    inline bool updateFromConfig(const Config& config) {
        changedValues.clear();
        for (auto& m : map)
            if (m.second.first(m.first, config))
                changedValues.insert(m.first);
        return !changedValues.empty();
    }

    inline void storeToConfig(Config config) {
        for (auto& m : map)
            m.second.second(m.first, config);
    }

    inline bool changedByName(const std::string& name) {
        return changedValues.count(name) > 0;
    }

    template<typename T>
    inline bool changed(const T& v) {
        auto it = changedMap.find((void*) &v);
        return it != changedMap.end() && changedValues.count(it->second);
    }
};
#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <boost/algorithm/string/predicate.hpp>
#include "glmHelpers.h"

class CameraPath {
public:
    struct Sample {
        float time = 0;
        glm::quat rot = glm::quat(0, 0, 0, 0);
        glm::vec3 pos = {0, 0, 0};

        Sample() = default;
        Sample(float t) : time(t) { };
        Sample(float t, const glm::mat4 &m) : time(t) { mat(m); }
        Sample(float t, const glm::quat &r, const glm::vec3 &p) : time(t), rot(r), pos(p) {}

        inline void mat(const glm::mat4 &m) {
            rot = glm::quat_cast(m);
            pos = glm::vec3(m[3]);
        }

        inline glm::mat4 mat() const {
            auto m = glm::mat4(glm::mat3_cast(rot));
            m[3] = glm::vec4(pos, 1);
            return m;
        }

        inline bool operator<(const Sample &s) {
            return time < s.time;
        }

        Sample mix(const Sample &s, float t) const {
            float f = (t - time) / (s.time - time);
            return Sample{t, glm::slerp(rot, s.rot, f), glm::mix(pos, s.pos, f)};
        }
    };

private:
    std::vector<Sample> samples;

public:

    inline float duration() const {
        return samples.empty() ? -1 : samples.back().time;
    }

    inline bool add(const Sample &s) {
        if (s.time < 0 || s.time < duration())
            return false;
        samples.push_back(s);
        return true;
    }

    inline void clear() { samples.clear(); }

    Sample sample(float t, bool cycle = false);

    bool load(std::istream &);
    bool loadBin(std::istream &);
    bool save(std::ostream &) const;

    inline bool loadFromFile(const std::string &file) {
        if (boost::algorithm::ends_with(file, ".bin")) {
            std::ifstream in(file, std::ios::binary);
            return in && loadBin(in);
        } else {
            std::ifstream in(file);
            return in && load(in);
        }
    }

    inline bool saveToFile(const std::string &file) const {
        std::ofstream out(file);
        return out && save(out);
    }
};
#include "CameraPath.h"
#include <algorithm>

using namespace std;
using namespace glm;

bool CameraPath::loadBin(istream& in){
    samples.clear();

    // alternative binary format
    int count = 0;
    in.read(reinterpret_cast<char*>(&count), sizeof(int));
    vector<mat4> matrices(count);
    vector<float> timestamps(count);
    in.read(reinterpret_cast<char*>(matrices.data()), sizeof(mat4) * count);
    in.read(reinterpret_cast<char*>(timestamps.data()), sizeof(float) * count);
    if (!in) return false;

    // convert to our format
    samples.resize(count);
    for (int i=0; i<count; i++) {
        samples[i].mat(rotate(mat4(1), pi<float>() / 2, {1, 0, 0}) * inverse(matrices[i])); // Y-up to Z-up
        samples[i].time = timestamps[i];
    }

    return true;
}

bool CameraPath::load(istream& in){
    samples.clear();
    Sample s;
    while(in >> s.time >> s.pos >> s.rot)
        samples.push_back(s);
    return (bool)in;
}

bool CameraPath::save(ostream& out) const {
    for(auto& s : samples)
        out << s.time << ' ' << s.pos << ' '<< s.rot << endl;
    return (bool)out;
}

CameraPath::Sample CameraPath::sample(float t, bool cycle) {
    if(samples.empty()) return Sample();
    if(cycle) {
        t = fmod(t, duration());
        if(t < 0) t += duration();
    }
    if(t <= samples.front().time) return samples.front();
    if(t >= samples.back().time) return samples.back();
    auto it = lower_bound (samples.begin(), samples.end(), Sample(t));
    return (it-1)->mix(*it, t);
}

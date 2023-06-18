#include "common.h"
#include <sstream>
#include <fstream>
#include <iomanip>

using namespace std;

string readFile(string file){
    ifstream fstream(file);
    stringstream stream;
    stream << fstream.rdbuf();
    return stream.str();
}

bool startsWith(const string& str, const string& start) {
    if (&start == &str) return true;
    if (start.length() > str.length()) return false;
    for (size_t i = 0; i < start.length(); ++i)
        if (start[i] != str[i]) return false;
    return true;
}

bool endsWith(const string& str, const string& end) {
    if (&end == &str) return true;
    if (end.length() > str.length()) return false;
    size_t offset = str.length() - end.length();
    for (size_t i = 0; i < end.length(); ++i)
        if (end[i] != str[i + offset]) return false;
    return true;
}

vector<string> split(string str, char delimiter) {
    vector<string> ret;
    stringstream ss(str);
    string s;

    while(getline(ss, s, delimiter))
        ret.push_back(s);

    return ret;
}

string to_string(double f, int num) {
    stringstream ss;
    ss << fixed << setprecision(num) << f;
    return ss.str();
}

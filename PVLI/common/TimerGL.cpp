#include "TimerGL.h"

using namespace std;
using namespace std::chrono;

TimerGL::~TimerGL() {
    TimerGL::reset();
}

void TimerGL::start(const std::string& what) {
    GLuint query;
    glGenQueries(1, &query);
    glQueryCounter(query, GL_TIMESTAMP);
    entries.push_back({what, 0});
    queries.push_back({query, 0});
    running = true;
}

void TimerGL::end() {
    if (!running) return;
    GLuint query;
    glGenQueries(1, &query);
    glQueryCounter(query, GL_TIMESTAMP);
    queries.back().second = query;
    running = false;
}

void TimerGL::finish() {
    end();
    for (int i = 0; i < queries.size(); i++) {
        GLuint64 t1, t2;
        auto& q = queries[i];
        if (!q.first) continue;

        glGetQueryObjectui64v(q.first, GL_QUERY_RESULT, &t1);
        glGetQueryObjectui64v(q.second ? q.second : queries[i + 1].first, GL_QUERY_RESULT, &t2);
        entries[i].elapsed = (double) (t2 - t1) / 1000;
        if (t1 > t2) entries[i].elapsed = 0; //FIXME better solution

        if (q.first) glDeleteQueries(1, &q.first);
        if (q.second) glDeleteQueries(1, &q.second);
        q.first = q.second = 0;
    }
    queries.clear();
}

void TimerGL::reset() {
    finish();
    entries.clear();
}

Timer::Entries TimerGL::getEntries() {
    finish();
    return entries;
}

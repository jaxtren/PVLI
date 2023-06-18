#version 430
layout(early_fragment_tests) in;

flat in int id;
layout(std430, binding = 0) buffer Mark {
	int mark[];
};

void main() {
	mark[id] = 1;
}

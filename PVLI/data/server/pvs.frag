#version 330

flat in int id;
layout(location = 0) out int pixel;

void main() {
	pixel = id;
}

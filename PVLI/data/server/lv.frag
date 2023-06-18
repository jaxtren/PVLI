#version 430

struct Fragment{
    uint color;
    float depth;
    int instID;
    int primID;
    int skip;
    int next;
};

layout (binding = 0) uniform sampler2D colorTex;
layout (binding = 0, r32i) uniform iimage2D head;

layout(std430, binding = 0) buffer Container {
    int count;
    Fragment fragments[];
} data;

uniform int instID;
flat in int primID;
in vec3 fnormal;
in vec2 fuv;

void main() {
    if (primID < 0) discard;

    //simple shading
    vec3 sunDir = normalize(vec3(1,-0.5,1));
    vec4 color = texture(colorTex, fuv) * mix(0.2, 1.0, clamp(dot(normalize(fnormal), sunDir), 0, 1));
    color = max(color, 1.0f/256.0f); //black color is considered as empty pixel

    //reserve space for fragment
    int offset = atomicAdd(data.count, 1);

    //link fragment
    Fragment fragment;
    fragment.next = imageAtomicExchange(head, ivec2(gl_FragCoord.xy), offset);
    fragment.color = packUnorm4x8(color);
    fragment.depth = gl_FragCoord.z;
    fragment.skip = 0;
    fragment.instID = instID;
    fragment.primID = primID;
    data.fragments[offset] = fragment;
}

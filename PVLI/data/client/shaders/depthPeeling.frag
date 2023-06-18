#version 330

uniform sampler2DArray prevDepth;
uniform float epsilon;
uniform int prevLayer;
uniform vec3 skipLayers; // factor, zNear, zFar

float linearizeDepth(float d) {
    return skipLayers.y * skipLayers.z / (skipLayers.z + d * (skipLayers.y - skipLayers.z));
}

float remapDepth(float d){
    return d + (1.0 - d) * epsilon;
}

void main() {
    if (prevLayer < 0) return;
    float depth = texelFetch(prevDepth, ivec3(gl_FragCoord.xy, prevLayer), 0).x;
    if (skipLayers.x > 0) {
        if (linearizeDepth(gl_FragCoord.z) <= linearizeDepth(depth) * (1.0f + skipLayers.x))
           discard;
    } else {
        if (gl_FragCoord.z <= remapDepth(depth))
            discard;
    }
}

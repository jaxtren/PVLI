#version 420
#define INVPI 0.31830988618379067153777f
#define INV2PI 0.15915494309189533576888f

in vec2 coord;
in vec3 dir;
layout(location = 0) out vec4 color;

// depth
layout(binding = 0) uniform sampler2D depth;

// color mapping
#ifdef REMOTE_TEXTURE_COLOR_MAPPING
uniform vec2 colorMapping;
vec4 applyColorMapping (vec4 c) {
    return (pow(vec4(colorMapping.x), c) - 1) * colorMapping.y;
}
#else
vec4 applyColorMapping(vec4 c) { return c; }
#endif

// gamma
#ifdef LOCAL_GAMMA
uniform float localGamma;
vec4 applyGamma(vec4 c) { return pow(c, vec4(localGamma)); }
#else
vec4 applyGamma(vec4 c) { return c; }
#endif

// local texture
#ifdef LOCAL_TEXTURE
layout(binding = 1) uniform sampler2D localColor;
#endif

// first remote texture
layout(binding = 2) uniform sampler2D remoteColor;
#ifdef REMOTE_TEXTURE_LAYER2
layout(binding = 5) uniform sampler2D remoteColorLayer2;
#endif
vec4 sampleRemoteColor() {
    vec4 c = texture(remoteColor, coord);
    #ifdef REMOTE_TEXTURE_LAYER2
    if (c.w == 0) c = texture(remoteColorLayer2, coord);
    #endif
    return c;
}

// second remote texture
#ifdef BLEND_SCENES
uniform float blendScenesFactor;
layout(binding = 3) uniform sampler2D remoteColor2;
#ifdef REMOTE_TEXTURE2_LAYER2
layout(binding = 6) uniform sampler2D remoteColor2Layer2;
#endif
vec4 sampleRemoteColor2() {
    vec4 c = texture(remoteColor2, coord);
    #ifdef REMOTE_TEXTURE2_LAYER2
    if (c.w == 0) c = texture(remoteColor2Layer2, coord);
    #endif
    return c;
}
#endif

// background
uniform vec3 backgroundColor;
#ifdef BACKGROUND_TEXTURE
layout(binding = 4) uniform sampler2D backgroundTexture;
vec4 sampleBackground(vec3 d) {
    float u = atan(d.y, d.x) * INV2PI;
    float v = acos(d.z) * INVPI;
    return textureLod(backgroundTexture, vec2(u, v), 0);
}
#endif

void main() {
    float d = texture(depth, coord).x;
    if (d < 1) {

        // remote color
        color = applyColorMapping(sampleRemoteColor());

        // second remote color
        #ifdef BLEND_SCENES
        vec4 color2 = applyColorMapping(sampleRemoteColor2());
        if (color2.w > 0)
            color = mix(color, color2, blendScenesFactor);
        #endif

        if (color.w == 0) color = vec4(1, 1, 1, 0);

        // local color
        #ifdef LOCAL_TEXTURE
        color *= applyGamma(vec4(texture(localColor, coord).xyz, 1));
        #endif
    }

    // background
    if (d == 1/* || color.w == 0*/) {
        color = vec4(backgroundColor, 1);
        #ifdef BACKGROUND_TEXTURE
        color *= sampleBackground(normalize(dir) * vec3(-1,1,1));
        #endif
        color = applyGamma(color);
    }

    color.w = 1;
}

#pragma once

#include "boostHelpers.h"
#include "glmHelpers.h"

/*
 * List of types and commands (messages) used in TCP communication
 * At the bottom: data structure of more complex messages
 */

ENUM_STRINGS_STREAM_OPS(RequestType, int, NONE,
    (NONE)
    (END)

    // client -> server (one way)
    (UPDATE_VIEWPOINT)
    (RENDER_REFERENCE)
    (SET_SETTINGS)
    (START_BENCHMARK)
    (RESET_STATE)
    (RELOAD_CONFIG)
    (RELOAD_SHADERS)

    // client -> server -> client (with response)
    (GET_SCENE_LIST)
    (GET_ALL_SETTINGS)
    (STOP_BENCHMARK)

    // server -> client (one way)
    (UPDATE_SCENE)
)

ENUM_STRINGS_STREAM_OPS(DataType, int, NONE,
    (NONE)
    (VIEW)
    (TEXTURE)
)

ENUM_STRINGS_STREAM_OPS(UpdateMode, int, DISABLE,
    (DISABLE)
    (ALLOW)
    (FORCE)
    (SERVER_ONLY) // render only on server, doesn't update PVS and doesn't send data using UPDATE_SCENE
                  // used for simulated benchmark to get higher quality of filter in lighthouse
)

struct ViewFlag {
    enum : int {
        VIDEO_STREAM = 1,
        SUBDIVIDE_CHECKER = 2,
        LAYER_FIRST_BLOCK_ORDER = 4,
        RELOCATED_PIXELS = 8,
        CUBEMAP = 16
    };
};

struct Viewpoint {
    enum Flag : int {
        AUTOMATIC_UPDATE = 1,
        SYNCHRONIZED = 2,
        DONT_SEND_PVS = 4,
    };

    glm::mat4 view = glm::mat4(0);
    int id = 0; // unique id of viewpoint, for multi-server synchronized updates is same for requests from all connections
                // always rising but doesn't have to be continual
    int flags = 0;
    float time = 0; // time since start in seconds
    float deltaTime = 0;  // predicted scene update delta time in seconds
    float latency = 0; // predicted scene update latency in seconds
    int videoBitrate = 0;
    int videoFramerate = 0;
};

namespace Debug {
    struct Vertex {
        glm::vec3 pos, color;
    };

    struct AABB {
        glm::vec3 min, max, color;
    };

    struct View {
        glm::mat4 transform;
        glm::mat4 projection;
        glm::vec3 color;
    };

    struct Sphere {
        glm::vec3 pos;
        float radius = 1;
        glm::vec3 color;
    };

    struct PVSCacheSample {
        glm::mat4 transform;
        glm::mat4 projection;
    };
}

/*

Rendering modes:
    illumination only: only illumination is streamed, albedo textures are stored on the client
    standard/classic: illumination with albedo is streamed (all color data)

Description of data structure for messages:

Material:
    string texture
    vec3 color
    bool opaque

UPDATE_VIEWPOINT (client -> server)
    Viewpoint
    UpdateMode

UPDATE_SCENE (server -> client):
    Viewpoint
    vec3 projParams // fovy, near, far
    ivec2 frameSize
    float gamma // for background and local textures in 'illumination only' mode
    vec2 colorMapping // for 'illumination only' mode, coefficients for logarithm color range mapping
    Material background
    int materialCount // 0: classic mode, otherwise: 'illumination only' mode, -1: reuse materials from previous update
    for materialCount:
        Material
    bool oneTimeVertices // vertices of whole scene are sent once at the beginning
    int vertexCount
    vector vertices // triangles * 3 vertices, using patches (VertexCompressor) or empty when using DONT_SEND_PVS
    while true:
        DataType type
        if type == NONE: break

        string name
        if type == VIEW:
            // base
            int flags // collection of ViewFlag
            int priority // for rendering and caching, (negative value - disable caching)
            mat4 projection
            mat4 view
            float blendFactor // factor for edge blending to have smooth transmission of shading between views
            int layerCount
            ivec2 layerSize
            ivec2 subdivide // subdivide layers to pixel blocks and render only interleaved parts based on subdivideOffset
                            // layerSize and fullLayersOffset have original/unsubdivided values
                            // layerSize is be multiple of subdivide
            ivec2 subdivideOffset
            float3 skipLayers // factor, zNear, zFar

            // full layers
            int fullLayers
            ivec2 fullLayersOffset // layer offset within texture
            string fullLayersTexture // may be empty if fullLayers == 0

            // blocks
            ivec3 blocksSize
            int blocksOffset // index of first tile
            string blocksTexture // may be empty if layerCount <= fullLayers
            vector blockCounts // number of layers per blocks (excluding full layers), may be empty if layerCount <= fullLayers
            vector blockIndices // index for every used tile in layer-first order or empty to compute indices in deterministic way
                                // in block-first or layer-first order based on ViewFlag::LAYER_FIRST_BLOCK_ORDER,
                                // if not empty also ViewFlag::LAYER_FIRST_BLOCK_ORDER is set and blocksOffset = 0

            // triangles, data are empty for: subdivide > {1,1} && subdivideOffset != {0, 0}
            int triangleCount
            vector triangleSubset // subset of triangles for shading
            vector triangleLayerMask // subset of triangles for every layer (for depth peeling optimization) or empty

        if type == TEXTURE:
            ivec2 size
            string stream // name of texture/stream from previous update, "raw", "jpeg" or empty
            if stream is empty: // new video stream
                VideoDecoderConfig videoConfig
            vector color
            vector mask // may be empty

    vector<Debug::AABB>
    vector<Debug::View>
    vector<Debug::Spheres>
    vector<Debug::Vertex> lines
    vector<Debug::PVSCacheSample>
    string stats
    float processingTime

    note:
        every video stream is referenced by max one view using videoStream

*/
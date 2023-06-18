#pragma once

#include <GL/glew.h>
#include <string>
#include <vector>
#include <memory>
#include "cudaHelpers.h"
#include "glmHelpers.h"
#include "glCudaHelpers.h"
#include "Culling.h"

#ifdef ENABLE_ASSIMP
#include <assimp/scene.h>
#endif

struct Material {
    glm::vec3 color = {1, 1, 1};
    GLuint texture = 0;
    glm::ivec2 textureSize = {0, 0};
    std::string textureName;
    int id = 0;
    bool opaque = true;

    inline ~Material() { free(); };
    void loadTexture(const std::string& file);
    void setTexture(glm::ivec2 size, void* data);
    void free();
};

struct Mesh {

    struct VertexData {
        glm::vec3 vertex;
        glm::vec3 normal;
        glm::vec2 uv;
        int material; // Lighthouse can have triangles with different materials in same mesh
    };

    struct Primitive {
        int size = 0;
        Material* material = nullptr;
    };

    inline ~Mesh() {
        glDeleteBuffers(1, &glVertices);
    }

    AABB aabb;
    int vertexCount = 0;
    GLuint glVertices = 0;
    CuBuffer<VertexData> cuVertices; // duplicate data to eliminate mapping between gl and cuda, because it's slow
    std::vector<Primitive> primitives;

    inline void setVertices(VertexData* data, int count) {
        vertexCount = (int)count;
        cuVertices.set(data, count);
        if (!glVertices) glGenBuffers(1, &glVertices);
        glBindBuffer(GL_ARRAY_BUFFER, glVertices);
        glBufferData(GL_ARRAY_BUFFER, count * sizeof(VertexData), (void*)data, GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        aabb.setNull();
        for (int i=0; i<count; i++)
            aabb.extend(data[i].vertex);
    }
};

class Scene {
public:

    struct Node {
        glm::mat4 transform = glm::mat4(1);
        std::vector<Node*> children;
        std::vector<Mesh*> meshes;
        ~Node();
    };

    struct Instance {
        glm::mat4 transform = glm::mat4(1);
        AABB aabb;
        Mesh* mesh = nullptr;
    };

public:
    Node* root = nullptr;
    std::vector<Mesh> meshes;
    std::vector<Material> materials;

#ifdef ENABLE_ASSIMP
    Node* createNode(aiNode*);
    bool load(const std::string& filePath);
#endif
    void collectForCameras(std::vector<Instance>&, const std::vector<glm::mat4>& cameras, Node*, glm::mat4);
    inline ~Scene() { free(); };
    void free();
    inline bool loaded() { return root; }

    /**
     * Collect instances visible from provided cameras or all instances
     * @param cameras list of cameras in form projection * view or empty to return all instances
     * @return collected instances
     */
    std::vector<Instance> collectForCameras(const std::vector<glm::mat4>& cameras);
};

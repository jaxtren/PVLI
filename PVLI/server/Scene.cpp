#include "Scene.h"

#include <iostream>
#include <functional>
#include <string>
#include "glmHelpers.h"
#include "stb_image.h"
#include "debug.h"

#ifdef ENABLE_ASSIMP
#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#endif

using namespace std;
using namespace glm;

void Material::loadTexture(const std::string& file) {
    textureName = file;
    int channels;
    stbi_set_flip_vertically_on_load(1);
    unsigned char* result = stbi_load(file.c_str(), &textureSize.x, &textureSize.y, &channels, 4);
    stbi_set_flip_vertically_on_load(0);
    if (!result) {
        cout << "Texture " << file << " cannot be loaded" << endl;
        return;
    }
    setTexture(textureSize, result);
    cout << "Loaded texture " << file << endl;
    stbi_image_free(result);
}

void Material::setTexture(glm::ivec2 size, void* data) {
    textureSize = size;
    free();

    // create texture
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, size.x, size.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    DEBUG_GPU_RAM << "glTexImage2D Material " << size.x * size.y * 4 << endl;
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    // detect opaque
    opaque = true;
    auto pixels = (unsigned char*) data;
    for (int i = 0; i < size.x * size.y; i++)
        if (pixels[i * 4 + 3] < 255) {
            opaque = false;
            break;
        }
}

void Material::free() {
    if (texture)
        glDeleteTextures(1, &texture);
    texture = 0;
}

Scene::Node::~Node() {
    for (auto n : children)
        delete n;
}

#ifdef ENABLE_ASSIMP
Scene::Node* Scene::createNode(aiNode* node) {
    Node* ret = new Node();
    ret->transform = glmMat4(node->mTransformation);
    for(unsigned i=0; i<node->mNumMeshes; i++)
        ret->meshes.push_back({&meshes[node->mMeshes[i]]});

    for(unsigned i=0; i<node->mNumChildren; i++)
        ret->children.push_back({createNode(node->mChildren[i])});

    return ret;
}

bool Scene::load(const std::string &filePath) {
    if (filePath.empty()) return false;
    free();

    // parse path
    string file = filePath, path;
    auto l = filePath.find_last_of("/\\");
    if (l != string::npos) {
        path = filePath.substr(0, l) + '/';
        file = filePath.substr(l + 1);
    }

    // scene
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile((path + file).c_str(),
                                             aiProcess_JoinIdenticalVertices | aiProcess_Triangulate);

    if (!scene) {
        cerr << importer.GetErrorString() << endl;
        return false;
    }

    cout << "Scene: " << filePath << endl;
    cout << "Materials: " << scene->mNumMaterials << endl;
    cout << "Meshes: " << scene->mNumMeshes << endl;

    // materials
    materials.resize(scene->mNumMaterials);
    for (unsigned i = 0; i < scene->mNumMaterials; i++) {
        auto src = scene->mMaterials[i];
        auto& dst = materials[i];
        dst.id = i;

        // color
        aiColor3D c;
        src->Get(AI_MATKEY_COLOR_DIFFUSE, c);
        dst.color = {c.r, c.g, c.b};

        // texture
        if (src->GetTextureCount(aiTextureType_DIFFUSE) > 0) {
            aiString aitex;
            src->Get(AI_MATKEY_TEXTURE(aiTextureType_DIFFUSE, 0), aitex);
            dst.loadTexture(path + aitex.C_Str());
        }

        // fallback: color -> texture
        if (!dst.texture) {
            auto color = u8vec4(dst.color * 255.0f, 255);
            dst.setTexture({1, 1}, &color);
        }
    }
    glBindTexture(GL_TEXTURE_2D, 0);

    // meshes
    meshes.resize(scene->mNumMeshes);
    for (unsigned i = 0; i < scene->mNumMeshes; i++) {
        auto src = scene->mMeshes[i];
        auto& dst = meshes[i];
        int mat = src->mMaterialIndex;
        vector<Mesh::VertexData> tmp, vertices;

        if (src->mTextureCoords[0] == nullptr)
            for (unsigned j = 0; j < src->mNumVertices; j++)
                tmp.push_back({glmVec3(src->mVertices[j]), glmVec3(src->mNormals[j]), {0, 0}, mat});
        else
            for (unsigned j = 0; j < src->mNumVertices; j++)
                tmp.push_back({glmVec3(src->mVertices[j]), glmVec3(src->mNormals[j]), glmVec2(src->mTextureCoords[0][j]), mat});

        for (unsigned j = 0; j < src->mNumFaces; j++) {
            auto& f = src->mFaces[j];
            if (f.mNumIndices != 3) continue;
            vertices.push_back(tmp[f.mIndices[0]]);
            vertices.push_back(tmp[f.mIndices[1]]);
            vertices.push_back(tmp[f.mIndices[2]]);
        }

        dst.setVertices(vertices.data(), (int)vertices.size());
        dst.primitives.push_back({(int)vertices.size(), &materials[mat]});
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // scene tree
    root = createNode(scene->mRootNode);

    return true;
}
#endif

void Scene::free() {
    delete root;
    root = nullptr;
    meshes.clear();
    materials.clear();
}

void Scene::collectForCameras(vector<Scene::Instance> & ret, const vector<glm::mat4>& cameras, Node *node, glm::mat4 transform) {
    auto t = transform * node->transform;
    for (auto& m : node->meshes)
        ret.push_back({t, m->aabb.transform(t), m});
    for (auto& n : node->children)
        collectForCameras(ret, cameras, n, t);
}

std::vector<Scene::Instance> Scene::collectForCameras(const std::vector<glm::mat4>& cameras) {
    vector<Scene::Instance> ret;
    if (root) collectForCameras(ret, cameras, root, mat4(1));
    return ret;
}

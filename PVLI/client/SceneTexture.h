#pragma once

#include <GL/glew.h>
#include <string>
#include <vector>
#include "px_sched.h"
#include "common.h"
#include "types.h"
#include "glmHelpers.h"
#include "BufferCache.h"
#include "DepthPeeling.h"
#include "asioHelpers.h"
#include "Timer.h"
#include "VideoCodingUtil.h"
#include "ffmpegDecoder.h"
#include "SceneData.h"

class SceneTexture : public SceneData {
public:
    glm::ivec2 size;
    std::string stream;
    VideoDecoderConfig videoConfig;
    std::unique_ptr<VideoDecoderBase> videoDecoder;
    std::vector<unsigned char> rawColor;
    std::vector<unsigned char> rawMask, rawMaskDecompressed;
    std::vector<std::vector<unsigned char>> rawMaskPartial;

    gl::Buffer textureBuf;
    GLuint texture = 0;

    std::string getStatsID();
    void process(SocketReader&);
    void beforeReuse();
    void free();

    void processJPEG();
    void generateTexture();

    bool useHwAccel() const;
    static std::unique_ptr<VideoDecoderBase> videoDecoderFactory(const VideoDecoderConfig& c, int w, int h, bool useHwAccel = false);

    struct {
        px_sched::Sync maskDecompressed, gpuResourcesReady, color, finish, gpuScale;
    } sync;

    private:
    void processHW(SocketReader&);
    void processSW(SocketReader&);

    void decompressMask(std::vector<std::vector<unsigned char>>& maskPartial, std::vector<unsigned char>& mask, std::vector<unsigned char>& maskDst, size_t maskSize);
};

#pragma once

#include <vector>

#include "VideoCodingUtil.h"

extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavutil/opt.h>
    #include <libswscale/swscale.h>
};

class ffmpegEncoder : public VideoEncoderBase
{
    AVCodecContext* context = nullptr;
    struct SwsContext * swsContext = nullptr;
    AVFrame* frame = nullptr;
    AVPacket* packet = nullptr;

    //VideoEncManager::CudaFrameToEncode inputFrame;
    std::vector<unsigned char> data;
public:
    ffmpegEncoder(const VideoEncoderConfig& c, int w, int h);
    ~ffmpegEncoder() override { destroy(); }

    bool init(const VideoEncoderConfig& c, int w, int h);
    void destroy();

    bool isInitialized() const { return context; }
    int getWidth() const { return isInitialized() ? context->width : 0; }
    int getHeight() const { return isInitialized() ? context->height : 0; }

    void prepareFrame(void* cudaSrcFrame, int w, int h) override;

	/**
     * encode RGB frame using h264 codec
     * @param data RGB buffer
     * @return encoded data
     */
    std::vector<unsigned char> encodeFrame(const VideoEncoderConfig& config) override;
};
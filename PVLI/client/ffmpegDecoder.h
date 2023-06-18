#pragma once

#include <vector>
#include "VideoCodingUtil.h"

extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavutil/hwcontext.h>
    #include <libswscale/swscale.h>

    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libavutil/pixdesc.h>
    #include <libavutil/hwcontext.h>
    #include <libavutil/opt.h>
    #include <libavutil/avassert.h>
    #include <libavutil/imgutils.h>
};

class ffmpegSwDecoder : public VideoDecoderBase
{
    AVCodecContext* context = nullptr;
    struct SwsContext* swsContext = nullptr;
    AVFrame* frame = nullptr;
    AVPacket* packet = nullptr;
    AVCodecParserContext* parser = nullptr;

public:
    ffmpegSwDecoder(const VideoDecoderConfig& c, int w, int h);
    ~ffmpegSwDecoder() override { destroy(); }


    bool init(const VideoDecoderConfig& c, int w, int h);
    void destroy();

    bool isHWAccelerated() override { return false; }
    bool decodeFrame(const std::vector<unsigned char>& data, void** decodedData) override;
};

class ffmpegHwDecoder : public VideoDecoderBase
{
    inline static enum AVPixelFormat hw_pix_fmt = AV_PIX_FMT_NONE;
    enum AVHWDeviceType device_type = AV_HWDEVICE_TYPE_NONE;
    AVCodecContext* context = nullptr;
    AVBufferRef* device_ctx = nullptr;
    struct SwsContext* swsContext = nullptr;
    AVFrame* frame = nullptr;
    AVFrame* device_frame = nullptr;
    AVPacket* packet = nullptr;
    AVCodecParserContext* parser = nullptr;

    int hw_decoder_init(AVCodecContext* ctx, AVHWDeviceType type);
    static AVPixelFormat get_hw_format(AVCodecContext* ctx, const AVPixelFormat* pix_fmts);
public:
    ffmpegHwDecoder(const VideoDecoderConfig& c, int w, int h);
    ~ffmpegHwDecoder() override { destroy(); }

    static bool queryHwAccel();

    bool init(const VideoDecoderConfig& c, int w, int h);
    void destroy();

    bool isHWAccelerated() override { return true; }
    bool decodeFrame(const std::vector<unsigned char>& data, void** decodedData) override;
};

#pragma once

#include "VideoCodingUtil.h"

#include <string>
#include <libavutil/pixfmt.h>

namespace ffmpegUtil
{
    inline AVPixelFormat getAVPixelFormat(const std::string& s)
    {
        if (s == "yuv444")
            return AV_PIX_FMT_YUV444P;

        return AV_PIX_FMT_YUV420P;
    }

    template <typename Config>
    inline AVPixelFormat getAVPixelFormat(const Config& c)
    {
        return getAVPixelFormat(c.format);
    }

    template <typename Config>
    inline AVCodecID getAVCodec(const Config& c)
    {
        if (c.codec == "hevc")
            return AV_CODEC_ID_HEVC;

        return AV_CODEC_ID_H264;
    }

    inline std::string getAVPreset(const VideoEncoderConfig& c)
    {
        if (c.preset <= 1)
            return "ultrafast";
        if (c.preset == 2)
            return "superfast";
        if (c.preset == 3)
            return "veryfast";
        if (c.preset == 4)
            return "faster";
        if (c.preset == 5)
            return "fast";
        if (c.preset == 6)
            return "medium";
        if (c.preset == 7)
            return "slow";
        if (c.preset == 8)
            return "slower";
        if (c.preset == 9)
            return "veryslow";

        return "placebo";
    }

    inline std::string getAVTuning(const VideoEncoderConfig& c)
    {
        if (c.tuning <= 1)
            return "zerolatency";
        if (c.tuning == 2)
            return "fastdecode";
        if (c.tuning == 3)
            return "ssim";
        if (c.tuning == 4)
            return "psnr";
        if (c.tuning == 5)
            return "stillimage";
        if (c.tuning == 6)
            return "grain";
        if (c.tuning == 7)
            return "animation";
        if (c.tuning == 8)
            return "film";

        return "placebo";
    }
}
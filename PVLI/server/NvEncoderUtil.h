#pragma once

#include "VideoCodingUtil.h"

#include <vector>
#include <nvEncodeAPI.h>

struct AVFrame;

namespace nvEncUtil
{
    inline NV_ENC_BUFFER_FORMAT getNvPixelFormat(const std::string& s)
    {
        if (s == "yuv444")
            return NV_ENC_BUFFER_FORMAT_YUV444;

        return NV_ENC_BUFFER_FORMAT_IYUV;
    }

	inline NV_ENC_BUFFER_FORMAT getNvPixelFormat(const VideoEncoderConfig& c)
    {
        return getNvPixelFormat(c.format);
    }

    inline GUID getNvCodec(const VideoEncoderConfig& c)
    {
        if (c.codec == "hevc")
            return NV_ENC_CODEC_HEVC_GUID;

        return NV_ENC_CODEC_H264_GUID;
    }

    inline GUID getNvPreset(const VideoEncoderConfig& c)
    {
        if (c.preset <= 1)
            return NV_ENC_PRESET_P1_GUID;
        if (c.preset == 2)
            return NV_ENC_PRESET_P2_GUID;
        if (c.preset == 3)
            return NV_ENC_PRESET_P3_GUID;
        if (c.preset == 4)
            return NV_ENC_PRESET_P4_GUID;
        if (c.preset == 5)
            return NV_ENC_PRESET_P5_GUID;
        if (c.preset == 6)
            return NV_ENC_PRESET_P6_GUID;

        return NV_ENC_PRESET_P7_GUID;
    }

    inline NV_ENC_TUNING_INFO getNvTuning(const VideoEncoderConfig& c)
    {
        if (c.tuning <= 1)
            return NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;
        if (c.tuning == 2)
            return NV_ENC_TUNING_INFO_LOW_LATENCY;
        if (c.tuning == 3)
            return NV_ENC_TUNING_INFO_HIGH_QUALITY;

        return NV_ENC_TUNING_INFO_LOSSLESS;
    }

    inline NV_ENC_PARAMS_RC_MODE getNvRateControl(const VideoEncoderConfig& c)
    {
        if (c.rateControl == "cbr")
            return NV_ENC_PARAMS_RC_CBR;
        if (c.rateControl == "vbr")
            return NV_ENC_PARAMS_RC_VBR;

        return NV_ENC_PARAMS_RC_CONSTQP;
    }
};

// frame is in device memory
class CudaFrameToEncode
{
public:
    std::string format;
    size_t cudaPitch = 0;
    int pitchLuma = 0;
    int pitchChroma = 0;
    std::vector<uint32_t> chromaOffsets;
    int width;
    int height;
    void* pDeviceFrame = nullptr;

    CudaFrameToEncode(int w, int h, std::string f);
    ~CudaFrameToEncode();
    CudaFrameToEncode(const CudaFrameToEncode&) = delete;
    CudaFrameToEncode& operator=(const CudaFrameToEncode&) = delete;
    CudaFrameToEncode(CudaFrameToEncode&& rhs) noexcept;
    CudaFrameToEncode& operator=(CudaFrameToEncode&& rhs) noexcept;

    void CopyDataToFrame(void* pSrcFrame, int width, int height);
    void CopyFrameToAVFrame(AVFrame* frame);
	    
private:
    void copyToYUV420(void* pSrcFrame, int width, int height);
    void copyToYUV444(void *pSrcFrame, int width, int height);
};
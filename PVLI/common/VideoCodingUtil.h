#pragma once

#include <string>
#include <vector>
#include "Config.h"
#include "asioHelpers.h"

struct VideoEncoderConfig
{
    std::string backend = "nv";
    int preset = 5;
    int tuning = 2;
    std::string codec = "hevc";
    std::string format = "yuv420";
    std::string rateControl = "vbr";
    int crfQuality = 25;
    int avgBitrate = 15'000'000;
    int maxBitrate = 50'000'000;
    int gopsize = 5;
    int refs = 1;
    int framerate = 10;

    bool multiPass = true;
    bool adaQuant = true;

    bool updateConfig(const Config& cfg)
    {
        bool ret =
            cfg.get("backend", backend) |
            cfg.get("codec", codec) |
            cfg.get("preset", preset) |
            cfg.get("tuning", tuning) |
            cfg.get("crfQuality", crfQuality) |
            cfg.get("avgBitrate", avgBitrate) |
            cfg.get("nvRateControl", rateControl) |
            cfg.get("nvMaxBitrate", maxBitrate) |
            cfg.get("nvMultiPass", multiPass) |
            cfg.get("nvAdaQuant", adaQuant) |
            cfg.get("gopsize", gopsize) |
            cfg.get("framerate", framerate) |
            cfg.get("refs", refs) |
            cfg.get("format", format);
        return ret;
    }

    void provideConfig(Config cfg)
    {
        cfg.set("backend", backend);
        cfg.set("codec", codec);
        cfg.set("preset", preset);
        cfg.set("tuning", tuning);
        cfg.set("crfQuality", crfQuality);
        cfg.set("avgBitrate", avgBitrate);
        cfg.set("nvRateControl", rateControl);
        cfg.set("nvMaxBitrate", maxBitrate);
        cfg.set("nvMultiPass", multiPass);
        cfg.set("nvAdaQuant", adaQuant);
        cfg.set("gopsize", gopsize);
        cfg.set("framerate", framerate);
        cfg.set("refs", refs);
        cfg.set("format", format);
    }
};

struct VideoDecoderConfig
{
    std::string codec = "hevc";
    std::string format = "yuv420";

    VideoDecoderConfig() = default;
    VideoDecoderConfig(const VideoEncoderConfig& cfg)
    {
        codec = cfg.codec;
        format = cfg.format;
    }

    bool updateConfig(const Config& cfg)
    {
        return
            cfg.get("codec", codec) |
            cfg.get("format", format);
    }

    void provideConfig(Config cfg)
    {
        cfg.set("codec", codec);
        cfg.set("format", format);
    }

    /* IO */

    template<typename S>
    inline void read(S& s)
    {
        s.read(codec);
        s.read(format);
    }

    template<typename S>
    inline void write(S& s) const
    {
        s.write(codec);
        s.write(format);
    }
};

class VideoEncoderBase
{
public:
    static constexpr int MIN_DIM_FOR_HW_ACCEL_DECODING = 144;
    static int getMinimalValidDim(int dim)
    {
        return std::max(MIN_DIM_FOR_HW_ACCEL_DECODING, dim);
    }

    VideoEncoderBase() = default;
    VideoEncoderBase(int w, int h) : width(w), height(h) {}
    virtual ~VideoEncoderBase() = default;

    // incoming data are expected to be in format PixelFormat::eR8G8B8u
    virtual void prepareFrame(void* cudaSrcFrame, int w, int h) = 0;
    virtual std::vector<unsigned char> encodeFrame(const VideoEncoderConfig& config) = 0;

    bool encoderReallocRequired(int w, int h) const
    {
        return width != w || height < h;
    }

    inline int getWidth() const { return width; }
    inline int getHeight() const { return height; }

protected:
    int width = 0;
    int height = 0;
};

class VideoDecoderBase
{
public:
    VideoDecoderBase() = default;
    VideoDecoderBase(int w, int h) : width(w), height(h) {}
    virtual ~VideoDecoderBase() = default;

    // destination data are expected to be in format PixelFormat::eR8G8B8A8u
    virtual bool isHWAccelerated() = 0;
    virtual bool decodeFrame(const std::vector<unsigned char>& data, void** decodedData) = 0;

//    bool encoderReallocRequired(int w, int h) const
//    {
//        return width != w || height < h;
//    }

    inline int getWidth() const { return width; }
    inline int getHeight() const { return height; }

protected:
    int width = 0;
    int height = 0;
};

struct AVFrame;
namespace hw_accel
{
    class Interop
    {
    public:
        virtual ~Interop() = default;

        virtual uint16_t GetGPUResources() = 0;
        virtual void PutBackGPUResources(uint16_t id) = 0;

        virtual void SubmitMask(uint16_t resId, int x, int y, unsigned char* mask) = 0;
        virtual void SubmitColor(uint16_t resId, AVFrame* avframe) = 0;
        virtual void ConversionAndMasking(uint16_t resId, unsigned target, int x, int y) = 0;
    };
}
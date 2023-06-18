#include "NvEncoderUtil.h"

#include <array>
#include <iostream>

#include "CustomConversions.h"
#include "NvEncoderCuda.h"

CudaFrameToEncode::CudaFrameToEncode(int w, int h, std::string f) : format(std::move(f)), width(w), height(h)
{
    const auto nvEncBufferFormat = nvEncUtil::getNvPixelFormat(format);
    const auto chromaCount = NvEncoder::GetNumChromaPlanes(nvEncBufferFormat);
    auto chromaHeight = chromaCount * NvEncoder::GetChromaHeight(nvEncBufferFormat, height);
    if (nvEncBufferFormat == NV_ENC_BUFFER_FORMAT_YV12 || nvEncBufferFormat == NV_ENC_BUFFER_FORMAT_IYUV)
        chromaHeight = NvEncoder::GetChromaHeight(nvEncBufferFormat, height);

    CUDA_DRVAPI_CALL(cuMemAllocPitch(reinterpret_cast<CUdeviceptr*>(&pDeviceFrame), &cudaPitch,
        NvEncoder::GetWidthInBytes(nvEncBufferFormat, width), height + chromaHeight, 16));

    pitchLuma = (int)cudaPitch;
    pitchChroma = NvEncoder::GetChromaPitch(nvEncBufferFormat, pitchLuma);
    NvEncoder::GetChromaSubPlaneOffsets(nvEncBufferFormat, (uint32_t)cudaPitch, height, chromaOffsets);
}

CudaFrameToEncode::~CudaFrameToEncode()
{
    try
    {
        if (pDeviceFrame)
            CUDA_DRVAPI_CALL(cuMemFree(reinterpret_cast<CUdeviceptr>(pDeviceFrame)));
    }
    catch (...)
    {
        std::cout << "CudaFrame free error." << std::endl;
    }

    pDeviceFrame = nullptr;
}

CudaFrameToEncode::CudaFrameToEncode(CudaFrameToEncode&& rhs) noexcept
{
    *this = std::move(rhs);
}

CudaFrameToEncode& CudaFrameToEncode::operator=(CudaFrameToEncode&& rhs) noexcept
{
    format = rhs.format;
    cudaPitch = rhs.cudaPitch;
    pitchLuma = rhs.pitchLuma;
    pitchChroma = rhs.pitchChroma;
    chromaOffsets = rhs.chromaOffsets;
    width = rhs.width;
    height = rhs.height;
    // take ownership of cuda resource
    pDeviceFrame = rhs.pDeviceFrame;
    rhs.pDeviceFrame = nullptr;

    return *this;
}

void CudaFrameToEncode::CopyDataToFrame(void* pSrcFrame, int width, int height)
{
    this->width = width;
    this->height = height;
    if (format == "yuv444")
        copyToYUV444(pSrcFrame, width, height);
    else
        copyToYUV420(pSrcFrame, width, height);
}

void CudaFrameToEncode::CopyFrameToAVFrame(AVFrame* frame)
{
    //const auto nvEncBufferFormat = FormatUtil::toNvCodecFormat(format);

    //CUDA_MEMCPY2D m = { 0 };
    //m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    //m.srcDevice = reinterpret_cast<CUdeviceptr>(pDeviceFrame);
    //m.srcPitch = pitchLuma;

    //m.dstMemoryType = CU_MEMORYTYPE_HOST;
    //m.dstHost = frame->data[0];
    //m.dstPitch = frame->linesize[0];

    //m.WidthInBytes = NvEncoder::GetWidthInBytes(nvEncBufferFormat, frame->width);
    //m.Height = frame->height;
    //CUDA_DRVAPI_CALL(cuMemcpy2D(&m));

    //uint32_t chromaHeight = NvEncoder::GetChromaHeight(nvEncBufferFormat, frame->height);
    //uint32_t chromaWidthInBytes = NvEncoder::GetChromaWidthInBytes(nvEncBufferFormat, frame->width);

    //for (uint32_t i = 0; i < chromaOffsets.size(); ++i)
    //    if (chromaHeight)
    //    {
    //        m.srcDevice = reinterpret_cast<CUdeviceptr>(static_cast<uint8_t*>(pDeviceFrame) + chromaOffsets[i]);
    //        m.srcPitch = pitchChroma;

    //        m.dstHost = frame->data[1 + i];
    //        m.dstPitch = frame->linesize[1 + i];

    //        m.WidthInBytes = chromaWidthInBytes;
    //        m.Height = chromaHeight;

    //        CUDA_DRVAPI_CALL(cuMemcpy2D(&m));
    //    }
}

void CudaFrameToEncode::copyToYUV420(void* pSrcFrame, int width, int height)
{
    const auto srcLineStep = 3 * width;
    std::array<int, 3> dstLineSteps = {pitchLuma, pitchChroma, pitchChroma };

    auto* dstFrame = static_cast<unsigned char*>(pDeviceFrame);
    std::array<unsigned char*, 3> dstFrames = { dstFrame, dstFrame + chromaOffsets[0], dstFrame + chromaOffsets[1] };

    cConv::RGBToYUV420_8u_C3P3R(pSrcFrame, srcLineStep, dstFrames.data(), dstLineSteps.data(), width, height);

    // TODO check npperror
    //auto* dstFrame = static_cast<Npp8u*>(pDeviceFrame);
    //auto* nppSrc = static_cast<Npp8u*>(pSrcFrame);
    //std::array<Npp8u*, 3> nppDst = { dstFrame, dstFrame + chromaOffsets[0], dstFrame + chromaOffsets[1] };
    //const auto nppROI = NppiSize{ width, height };
    //const auto ke2 = nppiGammaFwd_8u_C3IR(nppSrc, nppSrcLineStep, nppROI);
    //const auto ke = nppiRGBToYUV420_8u_C3P3R(nppSrc, nppSrcLineStep, nppDst.data(), nppDstLineStep.data(), nppROI);
}

void CudaFrameToEncode::copyToYUV444(void* pSrcFrame, int width, int height)
{
    const auto srcLineStep = 3 * width;
    std::array<int, 3> dstLineSteps = { pitchLuma, pitchChroma, pitchChroma };

    auto* dstFrame = static_cast<unsigned char*>(pDeviceFrame);
    std::array<unsigned char*, 3> dstFrames = { dstFrame, dstFrame + chromaOffsets[0], dstFrame + chromaOffsets[1] };

    cConv::RGBToYUV444_8u_C3P3R(pSrcFrame, srcLineStep, dstFrames.data(), dstLineSteps.data(), width, height);
}
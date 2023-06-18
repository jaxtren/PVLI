/*
* Copyright 2017-2020 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#include "NvEncoderCuda.h"

NvEncoderCuda::NvEncoderCuda(
    CUcontext cuContext, const VideoEncoderConfig& c, uint32_t nWidth, uint32_t nHeight,
    uint32_t nExtraOutputDelay, bool bMotionEstimationOnly, bool bOutputInVideoMemory) :
    NvEncoder(NV_ENC_DEVICE_TYPE_CUDA, cuContext, nWidth, nHeight, nvEncUtil::getNvPixelFormat(c), nExtraOutputDelay, bMotionEstimationOnly, bOutputInVideoMemory),
    VideoEncoderBase(nWidth, nHeight),
    m_cuContext(cuContext),
    pFormat(c.format)
{
    if (!m_hEncoder) 
    {
        NVENC_THROW_ERROR("Encoder Initialization failed", NV_ENC_ERR_INVALID_DEVICE);
    }

    if (!m_cuContext)
    {
        NVENC_THROW_ERROR("Invalid Cuda Context", NV_ENC_ERR_INVALID_DEVICE);
    }

    NV_ENC_INITIALIZE_PARAMS initializeParams = { NV_ENC_INITIALIZE_PARAMS_VER };
    NV_ENC_CONFIG encodeConfig = { NV_ENC_CONFIG_VER };
    initializeParams.encodeConfig = &encodeConfig;

    try
    {
        CreateDefaultEncoderParams(&initializeParams, c);
        CreateEncoder(&initializeParams);
        framerate = c.framerate;
        avgBitrate = c.avgBitrate;
        maxBitrate = c.maxBitrate;
    }
    catch (const NVENCException& e)
    {
        std::cout << e.what() << std::endl;
    }
}

NvEncoderCuda::~NvEncoderCuda()
{
    ReleaseCudaResources();
}

void NvEncoderCuda::prepareFrame(void* cudaSrcFrame, int w, int h)
{
    inputFrames[GetNextInputFrameID()].CopyDataToFrame(cudaSrcFrame, w, h);
}

std::vector<unsigned char> NvEncoderCuda::encodeFrame(const VideoEncoderConfig& config)
{
    const auto resolutionChanged = inputFrames[GetNextInputFrameID()].width != GetEncodeWidth() || inputFrames[GetNextInputFrameID()].height != GetEncodeHeight();
    if (resolutionChanged || framerate != config.framerate || avgBitrate != config.avgBitrate || maxBitrate != config.maxBitrate)
    {
        NV_ENC_RECONFIGURE_PARAMS pReconfigureParams{ NV_ENC_RECONFIGURE_PARAMS_VER };

        NV_ENC_CONFIG encodeConfig = { NV_ENC_CONFIG_VER };
        pReconfigureParams.reInitEncodeParams.encodeConfig = &encodeConfig;
        GetInitializeParams(&pReconfigureParams.reInitEncodeParams);

        if (resolutionChanged)
        {
            pReconfigureParams.reInitEncodeParams.encodeWidth = inputFrames[GetNextInputFrameID()].width;
            pReconfigureParams.reInitEncodeParams.encodeHeight = getMinimalValidDim(inputFrames[GetNextInputFrameID()].height);
            pReconfigureParams.forceIDR = 1;
        }
        if (framerate != config.framerate)
            pReconfigureParams.reInitEncodeParams.frameRateNum = framerate = config.framerate;
        if (avgBitrate != config.avgBitrate)
            encodeConfig.rcParams.averageBitRate = avgBitrate = config.avgBitrate;
        if (maxBitrate != config.maxBitrate)
            encodeConfig.rcParams.maxBitRate = maxBitrate = config.maxBitrate;

        Reconfigure(&pReconfigureParams);
    }

    std::vector<std::vector<uint8_t>> vPacket;
    EncodeFrame(vPacket);
    return vPacket.empty() ? std::vector<uint8_t>() : vPacket.back();
}

void NvEncoderCuda::AllocateInputBuffers(int32_t numInputBuffers)
{
    if (!IsHWEncoderInitialized())
    {
        NVENC_THROW_ERROR("Encoder intialization failed", NV_ENC_ERR_ENCODER_NOT_INITIALIZED);
    }

    // for MEOnly mode we need to allocate seperate set of buffers for reference frame
    int numCount = m_bMotionEstimationOnly ? 2 : 1;

    std::vector<void*> framesPtrs;
    for (auto i = 0; i < numInputBuffers; i++)
    {
        inputFrames.emplace_back(width, height, pFormat);
        framesPtrs.emplace_back(inputFrames.back().pDeviceFrame);

        RegisterInputResources(
            framesPtrs,
            NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
            GetMaxEncodeWidth(),
            GetMaxEncodeHeight(),
            static_cast<int>(inputFrames.back().cudaPitch),
            GetPixelFormat(),
            false);
    }

    framesPtrs.clear();
    if (m_bMotionEstimationOnly)
        for (auto i = 0; i < numInputBuffers; i++)
        {
            referenceFrames.emplace_back(width, height, pFormat);
            framesPtrs.emplace_back(referenceFrames.back().pDeviceFrame);

            RegisterInputResources(
                framesPtrs,
                NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
                GetMaxEncodeWidth(),
                GetMaxEncodeHeight(),
                static_cast<int>(referenceFrames.back().cudaPitch),
                GetPixelFormat(),
                true);
        }
}

void NvEncoderCuda::SetIOCudaStreams(NV_ENC_CUSTREAM_PTR inputStream, NV_ENC_CUSTREAM_PTR outputStream)
{
    NVENC_API_CALL(m_nvenc.nvEncSetIOCudaStreams(m_hEncoder, inputStream, outputStream));
}

void NvEncoderCuda::ReleaseInputBuffers()
{
    ReleaseCudaResources();
}

void NvEncoderCuda::ReleaseCudaResources()
{
    UnregisterInputResources();
    m_cuContext = nullptr;
}
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

#pragma once

#include <vector>
#include <stdint.h>
#include <cuda.h>

#include "NvEncoder.h"
#include "NvEncoderUtil.h"
#include "VideoCodingUtil.h"

#define CUDA_DRVAPI_CALL( call )                                                                                                 \
    do                                                                                                                           \
    {                                                                                                                            \
        CUresult err__ = call;                                                                                                   \
        if (err__ != CUDA_SUCCESS)                                                                                               \
        {                                                                                                                        \
            const char *szErrName = NULL;                                                                                        \
            cuGetErrorName(err__, &szErrName);                                                                                   \
            std::ostringstream errorLog;                                                                                         \
            errorLog << "CUDA driver API error " << szErrName ;                                                                  \
            throw NVENCException::makeNVENCException(errorLog.str(), NV_ENC_ERR_GENERIC, __FUNCTION__, __FILE__, __LINE__);      \
        }                                                                                                                        \
    }                                                                                                                            \
    while (0)

struct VideoEncoderConfig;
/**
*  @brief Encoder for CUDA device memory.
*/
class NvEncoderCuda : public NvEncoder, public VideoEncoderBase
{
public:
    NvEncoderCuda(CUcontext cuContext, const VideoEncoderConfig& c, uint32_t nWidth, uint32_t nHeight,
        uint32_t nExtraOutputDelay = 0, bool bMotionEstimationOnly = false, bool bOPInVideoMemory = false);
    virtual ~NvEncoderCuda();

    int framerate = 0, avgBitrate = 0, maxBitrate = 0;
    void prepareFrame(void* cudaSrcFrame, int w, int h) override;
    std::vector<unsigned char> encodeFrame(const VideoEncoderConfig& config) override;
    void SetIOCudaStreams(NV_ENC_CUSTREAM_PTR inputStream, NV_ENC_CUSTREAM_PTR outputStream);

  protected:
    /**
    *  @brief This function is used to release the input buffers allocated for encoding.
    *  This function is an override of virtual function NvEncoder::ReleaseInputBuffers().
    */
    virtual void ReleaseInputBuffers() override;

private:
    /**
    *  @brief This function is used to allocate input buffers for encoding.
    *  This function is an override of virtual function NvEncoder::AllocateInputBuffers().
    */
    virtual void AllocateInputBuffers(int32_t numInputBuffers) override;

private:
    /**
    *  @brief This is a private function to release CUDA device memory used for encoding.
    */
    void ReleaseCudaResources();

protected:
    CUcontext m_cuContext;
     
private:
    std::string pFormat;
    std::vector<CudaFrameToEncode> inputFrames;
    std::vector<CudaFrameToEncode> referenceFrames;
};
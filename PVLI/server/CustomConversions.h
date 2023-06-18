#pragma once

namespace cConv
{
    void RGBToGrey_inPlace(void* pImg, int imgStep, int width, int height);
    void RGBToYUV420_8u_C3P3R(void* pSrc, int nSrcStepBytes, unsigned char* pDst[3], int rDstStep[3], int width, int height);
    void RGBToYUV444_8u_C3P3R(void* pSrc, int nSrcStepBytes, unsigned char* pDst[3], int rDstStep[3], int width, int height);
}
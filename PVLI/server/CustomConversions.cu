#include "CustomConversions.h"

#include <cuda.h>
#include <glm/gtc/type_precision.hpp>

__global__ void rgb_2_grey(unsigned char* pImg, int imgStepInBytes, int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const auto pixOffset = y * imgStepInBytes + x * sizeof(glm::u8vec3);
    auto* rgbPtr = reinterpret_cast<glm::u8vec3*>(pImg + pixOffset);
    auto& rgb = *rgbPtr;
    const auto grey = static_cast<glm::uint8>(0.299f * static_cast<float>(rgb.r) + 0.587f * static_cast<float>(rgb.y) + 0.114f * static_cast<float>(rgb.z));
    rgb = glm::u8vec3(grey, grey, grey);
}

__global__ void RGBToYUV420_8u_C3P3R_kernel(
    unsigned char* pSrc, int nSrcStepBytes,
    unsigned char* yDst, unsigned char* uDst, unsigned char* vDst, int yStepBytes, int uvStepBytes,
    int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const auto pixOffset = y * nSrcStepBytes + x * sizeof(glm::u8vec3);
    auto* rgbPtr = reinterpret_cast<glm::u8vec3*>(pSrc + pixOffset);
    auto& rgb = *rgbPtr;

    const auto offset = y * yStepBytes + x;
    yDst[offset] = ((66 * rgb.r + 129 * rgb.g + 25 * rgb.b) >> 8) + 16;

    if (x % 2 == 0 && y % 2 == 0)
    {
        const int ch_x = x / 2;
        const int ch_y = y / 2;

        const auto uvOffset = ch_y * uvStepBytes + ch_x;
        uDst[uvOffset] = ((-38 * rgb.r + -74 * rgb.g + 112 * rgb.b) >> 8) + 128;
        vDst[uvOffset] = ((112 * rgb.r + -94 * rgb.g + -18 * rgb.b) >> 8) + 128;
    }
}

__global__ void RGBToYUV420_8u_C3P3R_Y_kernel(
    unsigned char* pSrc, int nSrcStepBytes,
    unsigned char* yDst, int yStepBytes,
    int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const auto pixOffset = y * nSrcStepBytes + x * sizeof(glm::u8vec3);
    auto* rgbPtr = reinterpret_cast<glm::u8vec3*>(pSrc + pixOffset);
    auto& rgb = *rgbPtr;

    const auto offset = y * yStepBytes + x;
    yDst[offset] = ((66 * rgb.r + 129 * rgb.g + 25 * rgb.b) >> 8) + 16;
}

__global__ void RGBToYUV420_8u_C3P3R_UV_kernel(
    unsigned char* pSrc, int nSrcStepBytes,
    unsigned char* uDst, unsigned char* vDst, int uvStepBytes,
    int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (2 * x >= width || 2 * y >= height)
        return;

    const auto pixOffset = 2 * y * nSrcStepBytes + 2 * x * sizeof(glm::u8vec3);
    auto* rgbPtr = reinterpret_cast<glm::u8vec3*>(pSrc + pixOffset);
    auto& rgb = *rgbPtr;

    const auto uvOffset = y * uvStepBytes + x;
    uDst[uvOffset] = ((-38 * rgb.r + -74 * rgb.g + 112 * rgb.b) >> 8) + 128;
    vDst[uvOffset] = ((112 * rgb.r + -94 * rgb.g + -18 * rgb.b) >> 8) + 128;
}

void cConv::RGBToGrey_inPlace(void* pImg, int imgStep, int width, int height)
{
    const int BS = 16;
    const dim3 blockSize(BS, BS);
    const dim3 gridSize((width / BS) + 1, (height / BS) + 1);

    rgb_2_grey <<< gridSize, blockSize >>> (static_cast<unsigned char *>(pImg), imgStep, width, height);

    cudaDeviceSynchronize();
}

void cConv::RGBToYUV420_8u_C3P3R(void *pSrc, int nSrcStepBytes, unsigned char *pDst[3], int rDstStep[3], int width, int height)
{
    const int BS = 16;
    const dim3 blockSize(BS, BS);
    const dim3 gridSize((width / BS) + 1, (height / BS) + 1);
    const dim3 gridSizeUV((width / BS / 2) + 1, (height / BS / 2) + 1);

    RGBToYUV420_8u_C3P3R_kernel <<< gridSize, blockSize >>> (static_cast<unsigned char*>(pSrc), nSrcStepBytes, pDst[0], pDst[1], pDst[2], rDstStep[0], rDstStep[1], width, height);
    //RGBToYUV420_8u_C3P3R_Y_kernel <<< gridSize, blockSize >>> (static_cast<unsigned char*>(pSrc), nSrcStepBytes, pDst[0], rDstStep[0], width, height);
    //RGBToYUV420_8u_C3P3R_UV_kernel <<< gridSizeUV, blockSize >>> (static_cast<unsigned char*>(pSrc), nSrcStepBytes, pDst[1], pDst[2], rDstStep[1], width, height);
}

__global__ void RGBToYUV444_8u_C3P3R_kernel(
    unsigned char* pSrc, int nSrcStepBytes,
    unsigned char* yDst, unsigned char* uDst, unsigned char* vDst, int yStepBytes, int uvStepBytes,
    int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const auto pixOffset = y * nSrcStepBytes + x * sizeof(glm::u8vec3);
    auto* rgbPtr = reinterpret_cast<glm::u8vec3*>(pSrc + pixOffset);
    auto& rgb = *rgbPtr;

    const auto offset = y * yStepBytes + x;
    yDst[offset] = ((66 * rgb.r + 129 * rgb.g + 25 * rgb.b) >> 8) + 16;

    const auto uvOffset = y * uvStepBytes + x;
    uDst[uvOffset] = ((-38 * rgb.r + -74 * rgb.g + 112 * rgb.b) >> 8) + 128;
    vDst[uvOffset] = ((112 * rgb.r + -94 * rgb.g + -18 * rgb.b) >> 8) + 128;
}

void cConv::RGBToYUV444_8u_C3P3R(void* pSrc, int nSrcStepBytes, unsigned char* pDst[3], int rDstStep[3], int width, int height)
{
    const int BS = 16;
    const dim3 blockSize(BS, BS);
    const dim3 gridSize((width / BS) + 1, (height / BS) + 1);
    const dim3 gridSizeUV((width / BS / 2) + 1, (height / BS / 2) + 1);

    RGBToYUV444_8u_C3P3R_kernel <<< gridSize, blockSize >>> (static_cast<unsigned char*>(pSrc), nSrcStepBytes, pDst[0], pDst[1], pDst[2], rDstStep[0], rDstStep[1], width, height);
}
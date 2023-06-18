#ifdef ENABLE_NVJPEG
#include "JpegEncoder.h"

#include <string>
#include <iostream>

#include "nvjpeg.h"

//TODO move
static const char *nvjpevGetErrorEnum(nvjpegStatus_t error) {
    switch (error) {
        case NVJPEG_STATUS_SUCCESS:
            return "NVJPEG_STATUS_SUCCESS";

        case NVJPEG_STATUS_NOT_INITIALIZED:
            return "NVJPEG_STATUS_NOT_INITIALIZED";

        case NVJPEG_STATUS_INVALID_PARAMETER:
            return "NVJPEG_STATUS_INVALID_PARAMETER";

        case NVJPEG_STATUS_BAD_JPEG:
            return "NVJPEG_STATUS_BAD_JPEG";

        case NVJPEG_STATUS_JPEG_NOT_SUPPORTED:
            return "NVJPEG_STATUS_JPEG_NOT_SUPPORTED";

        case NVJPEG_STATUS_ALLOCATOR_FAILURE:
            return "NVJPEG_STATUS_ALLOCATOR_FAILURE";

        case NVJPEG_STATUS_EXECUTION_FAILED:
            return "NVJPEG_STATUS_EXECUTION_FAILED";

        case NVJPEG_STATUS_ARCH_MISMATCH:
            return "NVJPEG_STATUS_ARCH_MISMATCH";

        case NVJPEG_STATUS_INTERNAL_ERROR:
            return "NVJPEG_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

void cudaHandleError(nvjpegStatus_t error, const char *file, int line) {
    if (error != NVJPEG_STATUS_SUCCESS) {
        std::cerr << "NVjpeg ERROR: " << nvjpevGetErrorEnum(error) << " in " << file << " at line " << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

static int dev_malloc(void **p, size_t s) { return (int)cudaMalloc(p, s); }
static int dev_free(void *p) { return (int)cudaFree(p); }

bool JpegEncoder::init() {
    nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};
    cuEC(nvjpegCreate(NVJPEG_BACKEND_GPU_HYBRID, &dev_allocator, &nvjpeg_handle));
    cuEC(nvjpegJpegStateCreate(nvjpeg_handle, &jpeg_state));
    cuEC(nvjpegEncoderStateCreate(nvjpeg_handle, &encoder_state, nullptr));
    cuEC(nvjpegEncoderParamsCreate(nvjpeg_handle, &encode_params, nullptr));

    cuEC(nvjpegEncoderParamsSetQuality(encode_params, 50, nullptr));
    cuEC(nvjpegEncoderParamsSetOptimizedHuffman(encode_params, 1, nullptr));
    cuEC(nvjpegEncoderParamsSetSamplingFactors(encode_params, NVJPEG_CSS_444, nullptr));
    return true;
}

void JpegEncoder::setQuality(int q) {
    cuEC(nvjpegEncoderParamsSetQuality(encode_params, q, nullptr));
}

void JpegEncoder::setFormat(const string& f) {
    nvjpegChromaSubsampling_t sampling = NVJPEG_CSS_444;
    if (f == "yuv444") sampling = NVJPEG_CSS_444;
    else if (f == "yuv444") sampling = NVJPEG_CSS_444;
    else if (f == "yuv422") sampling = NVJPEG_CSS_422;
    else if (f == "yuv420") sampling = NVJPEG_CSS_420;
    else if (f == "yuv440") sampling = NVJPEG_CSS_440;
    else if (f == "yuv411") sampling = NVJPEG_CSS_411;
    else if (f == "yuv410") sampling = NVJPEG_CSS_410;
    else return;
    cuEC(nvjpegEncoderParamsSetSamplingFactors(encode_params, sampling, nullptr));
}

void JpegEncoder::destroy() {
    cuEC(nvjpegEncoderParamsDestroy(encode_params));
    cuEC(nvjpegEncoderStateDestroy(encoder_state));
    cuEC(nvjpegJpegStateDestroy(jpeg_state));
    cuEC(nvjpegDestroy(nvjpeg_handle));
}

std::vector<unsigned char> JpegEncoder::encode(unsigned char* cuData, unsigned int w, unsigned int h) {
    std::vector<unsigned char> ret;
    if(w == 0 || h == 0) return ret;

    nvjpegImage_t imgdesc =
        {
            { cuData, nullptr, nullptr, nullptr },
            { w * 3, 0, 0, 0 }
        };

    cuEC(nvjpegEncodeImage(nvjpeg_handle, encoder_state, encode_params, &imgdesc, NVJPEG_INPUT_RGBI, w, h, nullptr));
    size_t length;
    cuEC(nvjpegEncodeRetrieveBitstream(nvjpeg_handle, encoder_state, nullptr, &length, nullptr));
    ret.resize(length);
    cuEC(nvjpegEncodeRetrieveBitstream(nvjpeg_handle, encoder_state, ret.data(), &length, nullptr));
    ret.resize(length);
    return ret;
}

bool JpegEncoder::decode(std::vector<unsigned char>& data, unsigned char* cuOutput, unsigned int w, unsigned int h){
    nvjpegImage_t imgdesc =
        {
            { cuOutput, nullptr, nullptr, nullptr },
            { w * 3, 0, 0, 0 }
        };

    cuEC(nvjpegDecode(nvjpeg_handle, jpeg_state, data.data(), data.size(), NVJPEG_OUTPUT_RGBI, &imgdesc, nullptr));
    return true;
}

#endif
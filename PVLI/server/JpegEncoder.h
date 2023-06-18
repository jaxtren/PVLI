#pragma once
#ifdef ENABLE_NVJPEG

#include <string>
#include <vector>

#include "cudaHelpers.h"
#include "glmHelpers.h"
#include "Config.h"
#include <nvjpeg.h>

class JpegEncoder {

private:
    nvjpegHandle_t nvjpeg_handle;
    nvjpegJpegState_t jpeg_state;
    nvjpegEncoderParams_t encode_params;
    nvjpegEncoderState_t encoder_state;

public:

    bool init();
    void destroy();

    void setQuality(int);
    void setFormat(const string&);

    std::vector<unsigned char> encode(unsigned char* cuData, unsigned int w, unsigned int h);
    bool decode(std::vector<unsigned char>& data, unsigned char* cuOutput,  unsigned int w, unsigned int h);
};

#endif
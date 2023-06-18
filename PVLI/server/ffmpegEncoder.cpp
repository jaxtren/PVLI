#include "ffmpegEncoder.h"

#include <string>
#include <vector>
#include <cstring>
#include <cassert>
#include "common.h"
#include "cudaHelpers.h"

#include "ffmpegUtil.h"
using namespace std;

ffmpegEncoder::ffmpegEncoder(const VideoEncoderConfig& c, int w, int h) : VideoEncoderBase(w, h)
{
    init(c, w, h);
}

bool ffmpegEncoder::init(const VideoEncoderConfig& c, int w, int h) {
    destroy();

    //codec
    //avcodec_register_all();
    const AVCodec* avcodec = avcodec_find_encoder(ffmpegUtil::getAVCodec(c));
    if (!avcodec)
        throw string_exception("Codec not found");

    //context
    context = avcodec_alloc_context3(avcodec);
    if (!context) throw string_exception("Could not allocate video codec context");
    context->bit_rate = c.avgBitrate;
    context->width = w; // resolution must be a multiple of two
    context->height = h;
    context->time_base = {1, c.framerate};
    context->framerate = { c.framerate, 1};
    context->gop_size = c.gopsize;
    context->max_b_frames = 0;
    context->pix_fmt = ffmpegUtil::getAVPixelFormat(c);
    context->thread_count = 0;
    context->refs = c.refs;

    av_opt_set(context->priv_data, "zerolatency", "1", 0); //for x264_nvenc (not working?)
    av_opt_set(context->priv_data, "tune", ffmpegUtil::getAVTuning(c).c_str(), 0);
    av_opt_set(context->priv_data, "preset", ffmpegUtil::getAVPreset(c).c_str(), 0);

    if(c.crfQuality >= 0)
        av_opt_set_int(context->priv_data, "crf", c.crfQuality, 0);

    if (avcodec_open2(context, avcodec, NULL) < 0)
        throw string_exception("Could not open codec");

    //packet
    packet = av_packet_alloc();
    if (!packet) throw string_exception("Cannot allocate packet");

    //frame
    frame = av_frame_alloc();
    if (!frame) throw string_exception("Could not allocate video frame");
    frame->format = context->pix_fmt;
    frame->width = context->width;
    frame->height = context->height;
    frame->pts = 0;
    if (av_frame_get_buffer(frame, 32) < 0)
        throw string_exception("Could not allocate the video frame data");

    //sws
    swsContext = sws_getContext(w, h, AV_PIX_FMT_RGB24, w, h,
        ffmpegUtil::getAVPixelFormat(c), 0, 0, 0, 0);

    //buffer
    data.resize(w * h * 3);

    return true;

}

void ffmpegEncoder::prepareFrame(void* cudaSrcFrame, int w, int h) {
    if (av_frame_make_writable(frame) < 0)
        throw string_exception("av_frame_make_writable error");

    //if (useCudaConvert)
    //{
    //    inputFrame.CopyDataToFrame(cudaSrcFrame);
    //    inputFrame.CopyFrameToAVFrame(frame);
    //}
    //else
    const size_t frameSize = w * h * 3;
    cuEC(cudaCopy(data.data(), static_cast<unsigned char*>(cudaSrcFrame), frameSize, cudaMemcpyDeviceToHost));
    memset(data.data() + frameSize, 0, data.size() - frameSize); // rest of image is black

    uint8_t* inData[1] = { static_cast<uint8_t*>(data.data()) };
    int inLinesize[1] = { frame->width * 3 }; // RGB stride
    sws_scale(swsContext, inData, inLinesize, 0, frame->height, frame->data, frame->linesize);
}

vector<unsigned char> ffmpegEncoder::encodeFrame(const VideoEncoderConfig& config) {
    vector<unsigned char> compressed;
    int ret = avcodec_send_frame(context, frame);
    if (ret < 0)
        throw string_exception("Error sending a frame for encoding");

    while (ret >= 0) {
        ret = avcodec_receive_packet(context, packet);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
        else if (ret < 0) throw string_exception("Error during encoding");

        //copy data
        if (packet->size > 0) {
            compressed.resize(compressed.size() + packet->size);
            memcpy(compressed.data() + compressed.size() - packet->size, packet->data, packet->size);
        }
        av_packet_unref(packet);
    }
    frame->pts++;
    return compressed;
}

void ffmpegEncoder::destroy() {
    if(context) avcodec_free_context(&context);
    if(frame) av_frame_free(&frame);
    if(packet) av_packet_free(&packet);
    if(swsContext) sws_freeContext(swsContext);
    context = nullptr;
    frame = nullptr;
    packet = nullptr;
    swsContext = nullptr;
}
#include "ffmpegDecoder.h"

#include <vector>
#include <cstring>
#include <cassert>
#include "common.h"

#include "ffmpegUtil.h"


using namespace std;

ffmpegSwDecoder::ffmpegSwDecoder(const VideoDecoderConfig& c, int w, int h) : VideoDecoderBase(w, h)
{
    init(c, w, h);
}

bool ffmpegSwDecoder::init(const VideoDecoderConfig& c, int w, int h)
{
    destroy();

    //codec
    //avcodec_register_all();
    const AVCodec* codec = avcodec_find_decoder(ffmpegUtil::getAVCodec(c));
    if (!codec) throw string_exception("Codec not found");

    //parser
    parser = av_parser_init(codec->id);
    if (!parser)  throw string_exception("Parser not found");
    parser->flags |= PARSER_FLAG_COMPLETE_FRAMES;

    //context
    context = avcodec_alloc_context3(codec);
    if (!context) throw string_exception("Could not allocate video codec context");
    context->width = w;
    context->height = h;
    context->pix_fmt = ffmpegUtil::getAVPixelFormat(c);
    context->max_b_frames = 0;
    context->thread_type = FF_THREAD_SLICE;
    context->thread_count = std::thread::hardware_concurrency();
    av_opt_set(context->priv_data, "tune", "zerolatency", 0);

    /* For some codecs, such as msmpeg4 and mpeg4, width and height
       MUST be initialized there because this information is not
       available in the bitstream. */

    if (avcodec_open2(context, codec, NULL) < 0)
        throw string_exception("Could not open codec");

    //packet
    packet = av_packet_alloc();
    if (!packet) throw string_exception("Cannot allocate packet");

    //frame
    frame = av_frame_alloc();
    if (!frame) throw string_exception("Could not allocate video frame");

    //sws
    swsContext = sws_getContext(w, h, ffmpegUtil::getAVPixelFormat(c), w, h, AV_PIX_FMT_RGBA, 0, 0, 0, 0);

    return true;
}

void ffmpegSwDecoder::destroy()
{
    if (context) avcodec_free_context(&context);
    if (frame) av_frame_free(&frame);
    if (packet) av_packet_free(&packet);
    if (parser) av_parser_close(parser);
    if (swsContext) sws_freeContext(swsContext);

    context = nullptr;
    frame = nullptr;
    packet = nullptr;
    parser = nullptr;
    swsContext = nullptr;
}

bool ffmpegSwDecoder::decodeFrame(const std::vector<unsigned char>& data, void** decodedData)
{
    bool hasFrame = false;
    int data_size = (int)data.size();
    auto data_ptr = (const uint8_t*)data.data();
    while (data_size > 0) {
        int ret = av_parser_parse2(parser, context, &packet->data, &packet->size, (const uint8_t*)data_ptr, data_size, AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
        if (ret < 0) throw string_exception("Error while parsing");
        data_ptr += ret;
        data_size -= ret;

        if (packet->size > 0) {
            if (avcodec_send_packet(context, packet) < 0)
                return false;
                //throw string_exception("Error sending a packet for decoding");

            while (ret >= 0) {
                ret = avcodec_receive_frame(context, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
                else if (ret < 0) throw string_exception("Error during decoding");

                //ffmpeg line size should be aligned
                int align = 32;
                int dest_ls = frame->width * 4; //RGBA
                int buf_ls = ((dest_ls - 1) / align + 1) * align;

                int outLinesize[1] = { buf_ls };
                uint8_t* outData[1] = { (uint8_t*)*decodedData };

                if (dest_ls == buf_ls && (std::uintptr_t)*decodedData % align == 0) {
                    //convert directly to destination
                    sws_scale(swsContext, frame->data, frame->linesize, 0, frame->height, outData, outLinesize);
                }
                else {
                    //convert
                    std::vector<unsigned char> buffer(frame->height * buf_ls + align); //FIXME enlarged, otherwise sws_scale crashes
                    outData[0] = (uint8_t*)buffer.data();
                    sws_scale(swsContext, frame->data, frame->linesize, 0, frame->height, outData, outLinesize);

                    //copy
                    for (int i = 0; i < frame->height; i++)
                        memcpy(static_cast<unsigned char*>(*decodedData) + dest_ls * i, buffer.data() + buf_ls * i, dest_ls);
                }
                hasFrame = true;
            }
        }
    }
    return hasFrame;
}

int ffmpegHwDecoder::hw_decoder_init(AVCodecContext* ctx, const enum AVHWDeviceType type)
{
    int err = 0;

    if ((err = av_hwdevice_ctx_create(&device_ctx, type, nullptr, nullptr, 0)) < 0)
    {
        fprintf(stderr, "Failed to create specified HW device.\n");
        return err;
    }
    ctx->hw_device_ctx = av_buffer_ref(device_ctx);

    return err;
}

enum AVPixelFormat ffmpegHwDecoder::get_hw_format(AVCodecContext* ctx, const enum AVPixelFormat* pix_fmts)
{
    for (auto p = pix_fmts; *p != -1; p++)
        if (*p == hw_pix_fmt)
            return *p;

    fprintf(stderr, "Failed to get HW surface format.\n");
    return AV_PIX_FMT_NONE;
}

ffmpegHwDecoder::ffmpegHwDecoder(const VideoDecoderConfig& c, int w, int h) : VideoDecoderBase(w, h)
{
    init(c, w, h);
}

bool ffmpegHwDecoder::queryHwAccel()
{
    std::set<AVHWDeviceType> supportedApi
    {
#ifdef DECODE_DXVA2
        AV_HWDEVICE_TYPE_DXVA2,
#endif
    };
    auto hwIter = AV_HWDEVICE_TYPE_NONE;

    while ((hwIter = av_hwdevice_iterate_types(hwIter)) != AV_HWDEVICE_TYPE_NONE)
        if (supportedApi.find(hwIter) != supportedApi.end())
            return true;
    return false;
}

bool ffmpegHwDecoder::init(const VideoDecoderConfig& c, int w, int h)
{
    destroy();

    //codec
    //avcodec_register_all();

    // find first available hw device
    //device_type = av_hwdevice_iterate_types(device_type);
    //if (device_type == AV_HWDEVICE_TYPE_NONE)
    //    return false;

    // TODO abstract behind platform guard
    device_type = AV_HWDEVICE_TYPE_DXVA2;

    const AVCodec* codec = avcodec_find_decoder(ffmpegUtil::getAVCodec(c));
    if (!codec) throw string_exception("Codec not found");

    //parser
    parser = av_parser_init(codec->id);
    if (!parser)  throw string_exception("Parser not found");
    parser->flags |= PARSER_FLAG_COMPLETE_FRAMES;

    for (int i = 0;; i++)
    {
        const AVCodecHWConfig* config = avcodec_get_hw_config(codec, i);
        if (!config)
        {
            fprintf(stderr, "Decoder %s does not support device type %s.\n",
                codec->name, av_hwdevice_get_type_name(device_type));
            return false;
        }
        if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX && config->device_type == device_type)
        {
            hw_pix_fmt = config->pix_fmt;
            break;
        }
    }

    context = avcodec_alloc_context3(codec);
    context->width = w;
    context->height = h;
    context->pix_fmt = ffmpegUtil::getAVPixelFormat(c);
    context->max_b_frames = 0;
    context->thread_type = FF_THREAD_SLICE;
    context->thread_count = std::thread::hardware_concurrency();
    av_opt_set(context->priv_data, "tune", "zerolatency", 0);
    context->get_format = get_hw_format;

    av_log_set_level(AV_LOG_VERBOSE);

    if (hw_decoder_init(context, device_type) < 0)
        return false;

    if (avcodec_open2(context, codec, nullptr) < 0)
        throw string_exception("Could not open hw codec");

    //packet
    packet = av_packet_alloc();
    if (!packet) throw string_exception("Cannot allocate packet");

    //frame
    frame = av_frame_alloc();
    device_frame = av_frame_alloc();
    if (!frame || !device_frame)
        throw string_exception("Could not allocate video frame");

    //sws
    swsContext = sws_getContext(w, h, AV_PIX_FMT_NV12, w, h, AV_PIX_FMT_RGBA, 0, nullptr, nullptr, nullptr);

    return true;
}

void ffmpegHwDecoder::destroy()
{
    if (context) avcodec_free_context(&context);
    if (device_ctx) av_buffer_unref(&device_ctx);
    if (frame) av_frame_free(&frame);
    if (device_frame) av_frame_free(&device_frame);
    if (packet) av_packet_free(&packet);
    if (parser) av_parser_close(parser);
    if (swsContext) sws_freeContext(swsContext);

    context = nullptr;
    device_ctx = nullptr;
    frame = nullptr;
    device_frame = nullptr;
    packet = nullptr;
    parser = nullptr;
    swsContext = nullptr;
}

bool ffmpegHwDecoder::decodeFrame(const std::vector<unsigned char>& data, void** decodedData)
{
    auto hasFrame = false;
    int data_size = (int)data.size();
    auto* data_ptr = static_cast<const uint8_t*>(data.data());
    while (data_size > 0)
    {
        int ret = av_parser_parse2(parser, context, &packet->data, &packet->size, static_cast<const uint8_t*>(data_ptr), data_size, AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
        if (ret < 0) throw string_exception("Error while parsing");
        data_ptr += ret;
        data_size -= ret;

        if (packet->size > 0) {
            if (avcodec_send_packet(context, packet) < 0)
                return false;
//                throw string_exception("Error sending a packet for decoding");

            while (ret >= 0)
            {
                ret = avcodec_receive_frame(context, device_frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                    break;
                if (ret < 0)
                    throw string_exception("Error during decoding");

                *decodedData = device_frame;

                // hw_frame is released in second call to this func, this shortcut seems to work, but might introduce some troubles
                return true;
                hasFrame = true;
            }
        }
    }
    return hasFrame;
}
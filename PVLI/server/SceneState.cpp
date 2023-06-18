#include "SceneState.h"
#include "SceneState.inl"
#include "Application.h"

using namespace glm;

void SceneState::packTexture(const string& name, ivec2 size, const VideoEncoderConfig& config, u8vec3* cuColor, unsigned char* cuMask) {
    auto& response = app->response;
    TimerCPU timer;
    stats.setPrefix("Time.Texture." + name);

    response.write(DataType::TEXTURE);
    response.write(name);

    if (config.backend == "raw") { // borrowing video config
        response.write(size);
        response.write(string("raw"));
        timer("RAW");
        vector<u8vec3> color(size.x * size.y);
        cuEC(cudaCopy(color, cuColor));
        timer();
        response.write(color);
    } else if (config.backend == "jpeg") { // borrowing video config for jpeg
        response.write(size);
        response.write(string("jpeg"));

#ifdef ENABLE_NVJPEG
        timer("JPEG");
        app->jpegEncoder.setQuality(config.crfQuality);
        app->jpegEncoder.setFormat(config.format);
        response.write(app->jpegEncoder.encode((unsigned char*) (cuColor), size.x, size.y));
        timer();
#else
        cerr << "JPEG not supported (compiled without nvJPEG)" << endl;
        response.write(std::vector<unsigned char>());
#endif

    } else { // video

        // manage stream
        auto streamName = name;
        streams.emplace_back(SceneState::Stream());
        auto stream = &streams.back();
        stream->name = streamName;
        auto prevStream = app->previousState ? app->previousState->findStream(streamName) : nullptr;

        // wait for previous frame finish previous
        if (prevStream) {
            app->scheduler.waitFor(prevStream->videoFinished); //TODO async wait and copy
            swap(stream->videoEncoder, prevStream->videoEncoder);
        }

        // recreate if need
        auto& video = stream->videoEncoder;
        if (!video || video->encoderReallocRequired(size.x, size.y)) {
            video.reset(); // first destroy current stream
            video = app->videoEncoderFactory(config, size.x, size.y);
            streamName = ""; // new stream
        }

        // size FIXME nvEncode changes frame size but FFmpeg doesn't
        auto realVideoSize = size;
        if(config.backend != "nv")
            realVideoSize = { video->getWidth(), video->getHeight() };
        response.write(realVideoSize);

        // stream
        response.write(streamName);
        if(streamName.empty())
            VideoDecoderConfig(config).write(response);

        // compress
        timer("Video.Copy");
        video->prepareFrame(static_cast<void*>(cuColor), size.x, size.y);
        timer();
        asyncVector([=] (auto prom) {
            prom->set_value(stream->videoEncoder->encodeFrame(config));
        }, &stream->videoFinished, "Video");

    }

    // mask
    if (cuMask) {
        timer("Mask.Copy");
        vector<unsigned char> mask(size.x * size.y);
        cuEC(cudaCopy(mask, cuMask));
        timer();
        auto block = app->lv->blocks.blockSize * app->compression.qt;
        auto parts = app->compression.qtParts;
        if (block > 0 && parts > 1) {
            // compress in parallel (multiple tasks) and send as multipart vector
            auto sharedMask = make_shared<vector<unsigned char>>(move(mask));
            response.write<int>(-parts); // part count
            for (int part=0; part<parts; part++)
                asyncVector([=] (auto prom){
                    prom->set_value(Compression::compress(*sharedMask, size.x, size.y, block, part, parts));
                }, "Mask*");
        } else
            // compress in one background task
            asyncVector([=, mask = move(mask)] (auto prom){
                if (block > 0) prom->set_value(Compression::compress(mask, size.x, size.y, block));
                else prom->set_value(app->compression.compress(Compression::Method::RLE1, mask));
            }, "Mask");
    } else asyncEmptyVector("Mask");

    // stats
    timer.finish();
    stats.add("", timer.getEntries());

    stats.setPrefix("Data.Texture." + name);
    stats.add("Width", size.x);
    stats.add("Height", size.y);
    stats.setPrefix();
}
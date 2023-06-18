#include "SceneTexture.h"
#include "Scene.h"
#include "Scene.inl"
#include "SceneData.inl"
#include "Application.h"
#include "asioHelpers.h"
#include "imguiHelper.h"
#include "graphic.h"
#include "ffmpegDecoder.h"
#include "compression/Compression.h"
#include "compression/QT.h"
#include "compression/RLE.h"

using namespace std;
using namespace glm;
using boost::asio::ip::tcp;

static inline bool hasRanges(ConstBuffer src) {
    auto head = Compression::header(src);
    return head.method == Compression::Method::RLE1 || head.method == Compression::Method::RLE8;
}

static inline vector<Compression::RLE::Range> decompressRanges(ConstBuffer src) {
    if(Compression::header(src).method == Compression::Method::RLE1)
        return Compression::RLE::decompressMaskRange(Compression::decompress(src, true));
    else return Compression::RLE::decompressRange(Compression::decompress(src, true));
}

std::string SceneTexture::getStatsID() { return "Texture." + name; }

void SceneTexture::processSW(SocketReader& reader) {
    timer("Base");
    reader.read(size);

    SceneTexture* reuse = scene->reuse ? scene->reuse->findTexture(name) : nullptr;
    size_t bufSize = size.x * size.y * 4;

    if (reuse) {
        if (bufSize <= reuse->textureBuf.size)
            swap(textureBuf, reuse->textureBuf);
        swap(texture, reuse->texture);
    }

    if (!textureBuf.size)
        run(GPUTask::OTHER, [=]() {
            textureBuf.allocate(GL_PIXEL_UNPACK_BUFFER, (size_t)(bufSize * scene->app->bufferSizeMultiple));
            textureBuf.map();
        }, &sync.gpuResourcesReady);

    reader.read(stream);
    if (stream.empty())
        videoConfig.read(reader);

    timer("Color");
    readCompressed(reader, rawColor, "Color", size.x * size.y * 3);
    if (stream == "raw") {
        runAfter(sync.gpuResourcesReady, [=] {
            // RGB -> RGBA
            auto src = reinterpret_cast<u8vec3*>(rawColor.data());
            auto dst = reinterpret_cast<u8vec4*>(textureBuf.data);
            for (int i=0; i<size.x * size.y; i++)
                dst[i] = u8vec4(src[i], 0);
        }, &sync.color, "RAW");
    } else if (stream == "jpeg") {
        runAfter(sync.gpuResourcesReady, [=] { processJPEG(); }, &sync.color, "JPEG");
    } else {
        scene->receivedVideoBytes += rawColor.size();
        SceneTexture* prevTex = scene->previous ? scene->previous->findTexture(stream) : nullptr;
        runAfter(prevTex ? prevTex->sync.color : px_sched::Sync(), [=] {
            if (prevTex) {
                swap(videoConfig, prevTex->videoConfig);
                swap(videoDecoder, prevTex->videoDecoder);
            }
            if (!videoDecoder || videoDecoder->isHWAccelerated())
                videoDecoder = videoDecoderFactory(videoConfig, size.x, size.y, false);
            assert(videoDecoder.get());
            runAfter(sync.gpuResourcesReady, [=] {
                if (!videoDecoder->decodeFrame(rawColor, reinterpret_cast<void**>(&textureBuf.data)))
                    videoDecoder.reset();
            }, &sync.color, "Video");
        }, &sync.color);
    }

    timer("Mask");
    readCompressed(reader, rawMaskPartial, "Mask", size.x * size.y);
    if (rawMaskPartial.size() == 1 && rawMaskPartial[0].empty())  // no mask
        runAfter(sync.color, [this]() {
            generateTexture();
        }, &sync.finish);
    else
        run([this]() {
            decompressMask(rawMaskPartial, rawMask, rawMaskDecompressed, size.x * size.y);
            runAfter(sync.color, [this]() {
                runAfter(sync.maskDecompressed, [this]() {
                    // apply mask
                    auto color = (u8vec4 *) textureBuf.data;
                    for (int i = 0, e = size.x * size.y; i < e; i++)
                        color[i].w = rawMaskDecompressed[i] << 7;
                    generateTexture();
                }, &sync.finish);
            }, &sync.finish);
        }, &sync.finish);
}

void SceneTexture::processHW(SocketReader& reader)
{
    auto hwAccelInterop = scene->app->hwAccelInterop;
    const auto GPUResId = hwAccelInterop->GetGPUResources();

    timer("Base");
    reader.read(size);

    SceneTexture* reuse = scene->reuse ? scene->reuse->findTexture(name) : nullptr;
    if (reuse)
        swap(texture, reuse->texture);

    reader.read(stream);
    if (stream.empty())
        videoConfig.read(reader);

    timer("Color");
    readCompressed(reader, rawColor, "Color", size.x * size.y * 3);
    scene->receivedVideoBytes += rawColor.size();

    SceneTexture* prevTex = scene->previous ? scene->previous->findTexture(stream) : nullptr;
    runAfter(prevTex ? prevTex->sync.color : px_sched::Sync(), [=] {
        if (prevTex) {
            swap(videoConfig, prevTex->videoConfig);
            swap(videoDecoder, prevTex->videoDecoder);
        }
        if (!videoDecoder || !videoDecoder->isHWAccelerated())
            videoDecoder = videoDecoderFactory(videoConfig, size.x, size.y, true);
        assert(videoDecoder.get());

        void* hw_AVFrame = nullptr;
        if (!videoDecoder->decodeFrame(rawColor, &hw_AVFrame))
            videoDecoder.reset();
        run(GPUTask::UPLOAD, [=]() {
            hwAccelInterop->SubmitColor(GPUResId, static_cast<AVFrame*>(hw_AVFrame));
        }, &sync.color, "Upload");
    }, &sync.color, "Video");

    timer("Mask");
    readCompressed(reader, rawMaskPartial, "Mask", size.x * size.y);
    runAfter(sync.color, [=]() {
        decompressMask(rawMaskPartial, rawMask, rawMaskDecompressed, size.x * size.y);
        runAfter(sync.maskDecompressed, [=]() {
            std::for_each(rawMaskDecompressed.begin(), rawMaskDecompressed.end(), [](auto& m) { if (m != 0) m = 255; });
            run(GPUTask::UPLOAD, [=]() {
                hwAccelInterop->SubmitMask(GPUResId, size.x, size.y, rawMaskDecompressed.data());
            }, &sync.gpuResourcesReady);
        }, &sync.gpuResourcesReady, "Mask");

        runAfter(sync.gpuResourcesReady, [=]() {
            run(GPUTask::COMPUTE, [=]() {
                if (!texture)
                {
                    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
                    glGenTextures(1, &texture);
                    glBindTexture(GL_TEXTURE_2D, texture);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, size.x, size.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
                    glBindTexture(GL_TEXTURE_2D, 0);
                }
                hwAccelInterop->ConversionAndMasking(GPUResId, texture, size.x, size.y);
                hwAccelInterop->PutBackGPUResources(GPUResId);
            }, &sync.finish);
        }, & sync.finish);
    }, & sync.finish);
}

void SceneTexture::decompressMask(std::vector<std::vector<unsigned char>>& maskPartial, std::vector<unsigned char>& mask, std::vector<unsigned char>& maskDst, size_t maskSize)
{
    maskDst.resize(maskSize);
    memset(maskDst.data(), 0, maskSize);

    size_t resSize = 0;
    for (const auto& part : maskPartial)
        resSize += part.size();
    if (resSize == 0)
        return;

    mask.reserve(resSize);
    for (const auto& part : maskPartial)
        mask.insert(mask.end(), part.begin(), part.end());

    int partCount = (int)maskPartial.size();
    int bitOffset = 0;

    ConstBuffer cBuffer(mask);
    cBuffer.read<Compression::Header>();

    for (auto i = 0; i < partCount; i++) {
        run([&, cBuffer, i, partCount, bitOffset]() {
            Compression::QT::decompress(cBuffer, maskDst, i, partCount, bitOffset);
        }, &sync.maskDecompressed, "Mask");

        if (bitOffset == 0)
            bitOffset -= (sizeof(Compression::Header) + sizeof(Compression::QT::Header)) * 8;
        bitOffset += (int)maskPartial[i].size() * 8;
    }
}

void SceneTexture::process(SocketReader& reader) {
    reader.read(name);
    if (useHwAccel())
        processHW(reader);
    else
        processSW(reader);
}

void SceneTexture::processJPEG() {
#ifdef ENABLE_TURBOJPEG
    auto decoder = scene->app->jpegDecoder.lock();
    int jpegSubsamp;
    glm::ivec2 tsize;
    tjDecompressHeader2(*decoder, rawColor.data(), (unsigned)rawColor.size(), &tsize.x, &tsize.y, &jpegSubsamp);
    tjDecompress2(*decoder, rawColor.data(), (unsigned)rawColor.size(), reinterpret_cast<unsigned char*>(textureBuf.data),
                  tsize.x, 0, tsize.y, TJPF_RGBA, 0); //flags: TJFLAG_FASTDCT
#else
    cerr << "JPEG not supported (compiled without TurboJPEG)" << endl;
#endif
};

void SceneTexture::generateTexture() {
    auto s = scene->app->uploadChunkSize;
    int Y = s <= 0 ? size.y : std::max(1, s / size.x);
    for (int y = 0; y < size.y; y += Y) {
        int h = std::min(Y, size.y - y);
        run(GPUTask::UPLOAD, [=]() {
            glActiveTexture(GL_TEXTURE0);
            if (y == 0) {
                textureBuf.unmap();
                if (!texture) {
                    glGenTextures(1, &texture);
                    glBindTexture(GL_TEXTURE_2D, texture);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                } else glBindTexture(GL_TEXTURE_2D, texture);

                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, h == size.y ? textureBuf : 0);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, size.x, size.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
                if (h == size.y) {
                    glBindTexture(GL_TEXTURE_2D, 0);
                    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
                    return;
                }
            } else glBindTexture(GL_TEXTURE_2D, texture);

            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, textureBuf);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, y, size.x, h, GL_RGBA, GL_UNSIGNED_BYTE, (void*) (intptr_t)(y * size.x * 4));
            glBindTexture(GL_TEXTURE_2D, 0);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        }, &sync.finish, "Upload", size.x * h * 4 / 1024);
    }
}

bool SceneTexture::useHwAccel() const
{
    if (scene->app->hwAccelInterop)
    {
        if (name != "Blocks" && scene->app->video.hwaccel.primCubeM)
            return true;
        if (name == "Blocks" && scene->app->video.hwaccel.blocks)
            return true;
    }
    return false;
}

std::unique_ptr<VideoDecoderBase> SceneTexture::videoDecoderFactory(const VideoDecoderConfig& c, int w, int h, const bool useHwAccel)
{
    if (useHwAccel)
        return std::make_unique<ffmpegHwDecoder>(c, w, h);
    return std::make_unique<ffmpegSwDecoder>(c, w, h);
}

void SceneTexture::beforeReuse() {
    if (!scene->app->reuseTextures) {
        glDeleteTextures(1, &texture);
        texture = 0;
    }
    if (!useHwAccel())
        textureBuf.map();
}

void SceneTexture::free() {
    glDeleteTextures(1, &texture);
    texture = 0;
}

# PVLI
Reference implementation for _PVLI: Potentially Visible Layered Image for Real-Time Ray Tracing_.

This project contains client and server implementation 
but path-tracing and denoising is implemented inside Lighthouse2 framework (in`../lighthouse2`).

#### Lighthouse2
Framework for real-time ray tracing (https://github.com/jbikker/lighthouse2).

We added modifications related to PVLI: extending pathtracing and SVGF filter to support multiple layers and PVS
(mostly done in module `RenderCore_Optix7Filter`, other modules are unsupported).

### Building
Libraries listed in the requirements need to be manually downloaded and configured in CMake.
Optional libraries are automatically ignored if not provided.

#### Requirements
Windows / Linux (tested on Manjaro and Ubuntu)\
CMake, C++17 (tested on Visual Studio 2022 and gcc12) \
OpenGL 4.5\
Boost: https://www.boost.org/ \
GLEW: https://github.com/nigels-com/glew \
GLM: https://github.com/g-truc/glm \
FFmpeg: https://ffmpeg.org/

***Client only:***\
GLFW 3: https://www.glfw.org/ \
TurboJPEG: https://libjpeg-turbo.org/ (optional)\
DXVA2 (windows only)

***Server only:***\
NVIDIA GPU\
GLFW 3 or EGL\
CUDA, NVENC: https://developer.nvidia.com/cuda-downloads \
nvJPEG: https://developer.nvidia.com/nvjpeg (optional)\
Assimp: https://github.com/assimp/assimp (optional)\
Modified Lighthouse2: `../lighthouse2`

#### CMake options
Common (type: bool, default true): `BUILD_SERVER` `BUILD_CLIENT` `BUILD_TESTS` \
Server only (type: bool, default: automatically detected): `ENABLE_GLFW` `ENABLE_EGL`

### Usage
Application is separated to two exacutables: `bin/client`, `bin/server`.
Each executable has own separated data folders: `/data/server` resp. `/data/client` 
and need to be started with working directory set to its data folder.
Order of starting doesn't matter.
Optional argument is relative path to configuration file (default is `config.cfg`).
Base configuration files are `data/*/config.cfg`, see them for further information.

Almost every parameter is configurable in runtime (from GUI or by reloading config),
except that server cannot handle changing/reloading of scene.
Server accepts max one client at the time. Both client and server should handle reconnecting.

Note: we don't have any encryption and authentification for network communication.

Client controls: 
- WASD / arrows: movement
- LMB + mouse: camera rotation
- SPACE, CONTROL: move up / down
- other: see `Application::keyCallback()` in `client/Application.cpp`

### Credits / Acknowledgments
Dear ImGui https://github.com/ocornut/imgui \
FSE https://github.com/Cyan4973/FiniteStateEntropy \
STB (stb_image) https://github.com/nothings/stb \
PX (px_sched) https://github.com/pplux/px
Catch https://github.com/catchorg/Catch2
And libraries listed in Requirements section.

This work was supported by the Czech Science Foundation (GA18-20374S), Research Center 
for Informatics (CZ.02.1.01/0.0/0.0/16\_019/0000765), and by the Grant Agency of the 
Czech Technical University in Prague, No SGS22/173/OHK3/3T/13.


### Authors
Jaroslav Kravec (kravec@fel.cvut.cz, kravec.jaroslav@gmail.com)\
Martin Káčerik (kacerma2@fel.cvut.cz)\
Jiri Bittner (bittner@fel.cvut.cz)



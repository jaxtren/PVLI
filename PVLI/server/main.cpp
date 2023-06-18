#define PX_SCHED_IMPLEMENTATION 1
#include "px_sched.h"

#ifndef ENABLE_LIGHTHOUSE
//lighthouse has implementation of std_image
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#endif

#include <iostream>
#include <memory>
#include "Application.h"

static void winCmd() {
#ifdef WIN32
    if (!AllocConsole())
        return;

	FILE* fDummy;
    freopen_s(&fDummy, "CONOUT$", "w", stdout);
    freopen_s(&fDummy, "CONOUT$", "w", stderr);
    freopen_s(&fDummy, "CONIN$", "r", stdin);
    std::cout.clear();
    std::clog.clear();
    std::cerr.clear();
    std::cin.clear();
#endif
}

int main(int argc, char **argv) {
	winCmd();
    try
    {
        auto app = std::make_unique<Application>(argc > 1 ? string(argv[1]) : "");
        app->run();
    }
    catch (const std::exception& e)
    {
        std::cout << "Fatal error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
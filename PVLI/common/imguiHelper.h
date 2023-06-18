#pragma once

#define IMGUI_IMPL_OPENGL_LOADER_GLEW

#include "imgui/imgui.h"
#include "imgui/misc/cpp/imgui_stdlib.h"
#include "imgui/examples/imgui_impl_glfw.h"
#include "imgui/examples/imgui_impl_opengl3.h"

namespace GUI {

    inline void ResetInput(){
        auto& io = ImGui::GetIO();
        io.MousePos = ImVec2(-FLT_MAX,-FLT_MAX);
        for (int i = 0; i < 5; i++) io.MouseDown[i] = false;
        io.MouseWheel = io.MouseWheelH = 0;
        io.KeyCtrl = io.KeyShift = io.KeyAlt = io.KeySuper = false;
        for (int i = 0; i < 512; i++) io.KeysDown[i] = false;
        for (int i = 0; i < ImGuiNavInput_COUNT; i++) io.NavInputs[i] = 0;
    }

    inline void NewFrame(bool resetInput = false){
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        if(resetInput) ResetInput();
        ImGui::NewFrame();
    }

    inline void Render(){
        ImGui::Render();
        auto& io = ImGui::GetIO();
        glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }
}
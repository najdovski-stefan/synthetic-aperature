#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"

#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iomanip>
#include <algorithm>

#include "SyntheticAperture.h"

std::string GenerateTimestampedFilename(const std::string& base_name, const std::string& extension) {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << base_name << "_" << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S") << "." << extension;
    return ss.str();
}

struct UIState {
    bool show_config_window = true;
    bool show_input_window = true;
    bool show_output_window = true;
    bool show_plot_window = true;
    bool show_properties_window = true;
    bool adding_template_mode = false;
    float zoom_input = 1.0f;
    float zoom_output = 1.0f;
    bool auto_fit_input = true;
    bool auto_fit_output = true;

    bool processing_in_progress = false;
    std::string last_process_message;
};

struct TextureManager {
    GLuint firstFrameTexture = 0;
    GLuint templateTexture = 0;
    GLuint syntheticTexture = 0;
    GLuint depthMapTexture = 0;
    bool needs_update = false;

    ~TextureManager() {
        cleanup();
    }

    void cleanup() {
        if (firstFrameTexture) { glDeleteTextures(1, &firstFrameTexture); firstFrameTexture = 0; }
        if (templateTexture) { glDeleteTextures(1, &templateTexture); templateTexture = 0; }
        if (syntheticTexture) { glDeleteTextures(1, &syntheticTexture); syntheticTexture = 0; }
        if (depthMapTexture) { glDeleteTextures(1, &depthMapTexture); depthMapTexture = 0; }
    }
};

void MatToTexture(const cv::Mat& mat, GLuint& texture) {
    if (mat.empty()) return;
    if (texture == 0) glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    cv::Mat rgb_mat;
    if (mat.channels() == 3) {
        cv::cvtColor(mat, rgb_mat, cv::COLOR_BGR2RGB);
    } else {
        cv::cvtColor(mat, rgb_mat, cv::COLOR_GRAY2RGB);
    }
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rgb_mat.cols, rgb_mat.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb_mat.data);
}

float CalculateFitZoom(const cv::Mat& image, const ImVec2& available_size) {
    if (image.empty()) return 1.0f;
    float zoom_x = available_size.x / image.cols;
    float zoom_y = available_size.y / image.rows;
    return std::min(zoom_x, zoom_y);
}

bool IsMousePosValid(const cv::Point2i& pos, const cv::Mat& frame, int template_size) {
    if (frame.empty()) return false;
    return pos.x >= 0 && pos.y >= 0 &&
           pos.x + template_size <= frame.cols &&
           pos.y + template_size <= frame.rows;
}


void RenderConfigWindow(SyntheticAperture& processor, SA_Parameters& params, UIState& ui_state, TextureManager& textures) {
    if (!ui_state.show_config_window) return;

    static bool first_show = true;
    if (first_show) {
        ImGui::SetNextWindowPos(ImVec2(20, 40));
        ImGui::SetNextWindowSize(ImVec2(350, 750));
        first_show = false;
    }

    ImGui::Begin("Configuration", &ui_state.show_config_window);

    ImGui::SeparatorText("Video Input");
    static char videoPathBuf[1024] = "/Users/user/Downloads/IMG_2116.MOV";
    ImGui::InputText("##VideoPath", videoPathBuf, sizeof(videoPathBuf));

    if (ImGui::Button("Load Video", ImVec2(-1, 0))) {
        if (processor.loadVideo(videoPathBuf, params.max_frames, params.scale_factor)) {
            MatToTexture(processor.getFirstColorFrame(), textures.firstFrameTexture);
            params.template_points.clear();
            ui_state.last_process_message = "Video loaded. Add templates to begin.";
        }
    }

    ImGui::SeparatorText("Processing Parameters");
    ImGui::InputInt("Max Frames", &params.max_frames, 1, 10);
    params.max_frames = std::max(1, params.max_frames);
    ImGui::InputInt("Scale Factor", &params.scale_factor, 1, 2);
    params.scale_factor = std::max(1, params.scale_factor);
    ImGui::InputInt("Template Size", &params.template_size, 1, 5);
    params.template_size = std::max(10, params.template_size);
    ImGui::InputInt("Search Window", &params.search_window_size, 1, 5);
    params.search_window_size = std::max(params.template_size + 10, params.search_window_size);

    ImGui::SeparatorText("Depth Map Templates");
    ImVec4 button_color = ui_state.adding_template_mode ? ImVec4(0.8f, 0.3f, 0.3f, 1.0f) : ImVec4(0.26f, 0.59f, 0.98f, 1.0f);

    bool was_adding_mode = ui_state.adding_template_mode;
    if (was_adding_mode) {
        ImGui::PushStyleColor(ImGuiCol_Button, button_color);
    }

    if (ImGui::Button(was_adding_mode ? "Cancel Adding" : "Add Templates...", ImVec2(-1, 0))) {
        if (processor.isVideoLoaded()) {
            ui_state.adding_template_mode = !ui_state.adding_template_mode;
        }
    }

    if (was_adding_mode) {
        ImGui::PopStyleColor();
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Activate this mode, then click on the 'Input Frame' window to add template points.");


    if (ImGui::Button("Clear All Templates", ImVec2(-1, 0))) {
        params.template_points.clear();
    }
    ImGui::Text("Templates added: %zu", params.template_points.size());

    ImGui::BeginChild("TemplateList", ImVec2(0, 100), true, ImGuiWindowFlags_None);
    for(size_t i = 0; i < params.template_points.size(); ++i) {
        ImGui::Text("  %zu: (%d, %d)", i+1, params.template_points[i].x, params.template_points[i].y);
    }
    ImGui::EndChild();

    ImGui::SeparatorText("Processing & Output");
    bool can_process = processor.isVideoLoaded() && params.template_points.size() >= 2;
    if (!can_process || ui_state.processing_in_progress) ImGui::BeginDisabled();

    if (ImGui::Button(ui_state.processing_in_progress ? "PROCESSING..." : "PROCESS", ImVec2(-1, 40))) {
        ui_state.processing_in_progress = true;
        ui_state.last_process_message = "Processing...";
        if (processor.process(params)) {
            textures.needs_update = true;
            ui_state.last_process_message = "✓ Processing completed successfully!";
        } else {
            ui_state.last_process_message = "⚠ Processing failed: " + processor.getStatusMessage();
        }
        ui_state.processing_in_progress = false;
    }
    if (!can_process || ui_state.processing_in_progress) {
        ImGui::EndDisabled();
        if (!processor.isVideoLoaded()) ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), "Load video first");
        else if (params.template_points.size() < 2) ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), "Add at least 2 templates");
    }

    if (!ui_state.last_process_message.empty()) {
        ImGui::Separator();
        if (ui_state.last_process_message.find("✓") != std::string::npos) {
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "%s", ui_state.last_process_message.c_str());
        } else if (ui_state.last_process_message.find("⚠") != std::string::npos) {
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), "%s", ui_state.last_process_message.c_str());
        } else {
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "%s", ui_state.last_process_message.c_str());
        }
    }

    ImGui::End();
}

void RenderPropertiesWindow(SyntheticAperture& processor, SA_Parameters& params, UIState& ui_state) {
    if (!ui_state.show_properties_window) return;

    static bool first_show = true;
    if (first_show) {
        ImGui::SetNextWindowPos(ImVec2(390, 40));
        ImGui::SetNextWindowSize(ImVec2(300, 400));
        first_show = false;
    }

    ImGui::Begin("Properties", &ui_state.show_properties_window);

    ImGui::SeparatorText("Status");
    ImGui::TextWrapped("%s", processor.getStatusMessage().c_str());

    if (ui_state.adding_template_mode) {
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "🎯 Template adding mode active");
        ImGui::Text("Click on the input frame to add templates.");
    }

    ImGui::SeparatorText("Input View");
    ImGui::Checkbox("Auto Fit Input", &ui_state.auto_fit_input);
    if (!ui_state.auto_fit_input) ImGui::SliderFloat("Input Zoom", &ui_state.zoom_input, 0.1f, 5.0f, "%.1fx");

    ImGui::SeparatorText("Output View");
    ImGui::Checkbox("Auto Fit Output", &ui_state.auto_fit_output);
    if (!ui_state.auto_fit_output) ImGui::SliderFloat("Output Zoom", &ui_state.zoom_output, 0.1f, 5.0f, "%.1fx");

    if (processor.isProcessed()) {
        ImGui::SeparatorText("Processing Results");
        ImGui::Text("Templates processed: %zu", params.template_points.size());
    }
    ImGui::End();
}

void RenderInputWindow(SyntheticAperture& processor, SA_Parameters& params, UIState& ui_state, TextureManager& textures) {
    if (!ui_state.show_input_window) return;

    static bool first_show = true;
    if (first_show) { ImGui::SetNextWindowPos(ImVec2(710, 40)); ImGui::SetNextWindowSize(ImVec2(600, 500)); first_show = false; }

    ImGui::Begin("Input Frame", &ui_state.show_input_window);

    if (!processor.isVideoLoaded()) { ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "No video loaded"); ImGui::End(); return; }

    cv::Mat frame = processor.getFirstColorFrame();
    ImVec2 available_size = ImGui::GetContentRegionAvail();
    float zoom = ui_state.auto_fit_input ? CalculateFitZoom(frame, available_size) : ui_state.zoom_input;
    if (ui_state.auto_fit_input) ui_state.zoom_input = zoom;

    ImGui::BeginChild("InputScroll", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);
    ImGui::Image((void*)(intptr_t)textures.firstFrameTexture, ImVec2(frame.cols * zoom, frame.rows * zoom));

    cv::Point2i hover_pos = {-1, -1};
    if (ImGui::IsItemHovered()) {
        ImVec2 mouse_pos = ImGui::GetMousePos();
        ImVec2 image_min = ImGui::GetItemRectMin();
        hover_pos.x = (int)((mouse_pos.x - image_min.x) / zoom);
        hover_pos.y = (int)((mouse_pos.y - image_min.y) / zoom);

        if (ui_state.adding_template_mode) {
            ImGui::SetTooltip("Click to add template at (%d, %d)", hover_pos.x, hover_pos.y);
            if (ImGui::IsMouseClicked(0)) {
                if (IsMousePosValid(hover_pos, frame, params.template_size)) {
                    params.template_points.push_back(hover_pos);
                } else {
                    ui_state.last_process_message = "⚠ Template position is out of bounds!";
                }
            }
        } else {
            ImGui::SetTooltip("Frame: %d x %d, Zoom: %.1fx\nPosition: (%d, %d)", frame.cols, frame.rows, zoom, hover_pos.x, hover_pos.y);
        }
    }

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 image_min = ImGui::GetItemRectMin();

    for (size_t i = 0; i < params.template_points.size(); ++i) {
        const auto& p = params.template_points[i];
        ImVec2 rect_min = { image_min.x + p.x * zoom, image_min.y + p.y * zoom };
        ImVec2 rect_max = { rect_min.x + params.template_size * zoom, rect_min.y + params.template_size * zoom };
        draw_list->AddRect(rect_min, rect_max, IM_COL32(0, 255, 255, 255), 1.0f, 0, 2.0f); // Cyan
        char buf[16];
        snprintf(buf, 16, "%zu", i + 1);
        draw_list->AddText({rect_max.x + 4, rect_min.y}, IM_COL32(255, 255, 255, 255), buf);
    }

    if (ui_state.adding_template_mode && IsMousePosValid(hover_pos, frame, params.template_size)) {
        ImVec2 hover_rect_min = { image_min.x + hover_pos.x * zoom, image_min.y + hover_pos.y * zoom };
        ImVec2 hover_rect_max = { hover_rect_min.x + params.template_size * zoom, hover_rect_min.y + params.template_size * zoom };
        draw_list->AddRect(hover_rect_min, hover_rect_max, IM_COL32(255, 255, 0, 200), 1.0f, 0, 1.5f);
    }

    ImGui::EndChild();
    ImGui::End();
}

void RenderOutputWindow(SyntheticAperture& processor, UIState& ui_state, TextureManager& textures) {
    if (!ui_state.show_output_window) return;

    static bool first_show = true;
    if (first_show) { ImGui::SetNextWindowPos(ImVec2(710, 560)); ImGui::SetNextWindowSize(ImVec2(600, 400)); first_show = false; }

    ImGui::Begin("Output Results", &ui_state.show_output_window);

    if (!processor.isProcessed()) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "No processing results");
        if (ui_state.processing_in_progress) {
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "⏳ Processing in progress...");
        }
        ImGui::End();
        return;
    }

    ImGui::BeginTabBar("OutputTabs");

    if (ImGui::BeginTabItem("Depth Map")) {
        cv::Mat depth_map = processor.getDepthMap();
        if (!depth_map.empty()) {
            if (ImGui::Button("Save Depth Map")) {
                std::string filename = GenerateTimestampedFilename("depth_map", "png");
                if (cv::imwrite(filename, depth_map)) {
                    ui_state.last_process_message = "✓ Saved " + filename;
                } else {
                    ui_state.last_process_message = "⚠ Failed to save " + filename;
                }
            }
            ImGui::SameLine();

            ImVec2 available_size = ImGui::GetContentRegionAvail();

            ImGui::Text("Depth Legend:");
            ImDrawList* draw_list = ImGui::GetWindowDrawList();
            ImVec2 p = ImGui::GetCursorScreenPos();
            float legend_width = 200.0f;
            float legend_height = 20.0f;
            draw_list->AddRectFilledMultiColor(p, ImVec2(p.x + legend_width, p.y + legend_height), IM_COL32(0,0,255,255), IM_COL32(255,0,0,255), IM_COL32(255,0,0,255), IM_COL32(0,0,255,255));
            ImGui::SetCursorScreenPos({p.x, p.y + legend_height + 2});
            ImGui::Text("Furthest (Blue)");
            ImGui::SameLine(p.x + legend_width - ImGui::CalcTextSize("Nearest (Red)").x);
            ImGui::Text("Nearest (Red)");
            ImGui::Dummy(ImVec2(0, legend_height + 5));

            float zoom = ui_state.auto_fit_output ? CalculateFitZoom(depth_map, available_size) : ui_state.zoom_output;
            if (ui_state.auto_fit_output) ui_state.zoom_output = zoom;

            ImGui::Image((void*)(intptr_t)textures.depthMapTexture, ImVec2(depth_map.cols * zoom, depth_map.rows * zoom));
        } else {
             ImGui::Text("Depth map not generated. Process with >= 2 templates.");
        }
        ImGui::EndTabItem();
    }

    if (ImGui::BeginTabItem("Synthetic Aperture")) {
        cv::Mat synthetic_img = processor.getSyntheticImage();
        if (!synthetic_img.empty()) {
            if (ImGui::Button("Save Synthetic Image")) {
                std::string filename = GenerateTimestampedFilename("synthetic_aperture", "png");
                if (cv::imwrite(filename, synthetic_img)) {
                    ui_state.last_process_message = "✓ Saved " + filename;
                } else {
                    ui_state.last_process_message = "⚠ Failed to save " + filename;
                }
            }
            ImGui::SameLine();

            ImVec2 available_size = ImGui::GetContentRegionAvail();
            float zoom = ui_state.auto_fit_output ? CalculateFitZoom(synthetic_img, available_size) : ui_state.zoom_output;
            if (ui_state.auto_fit_output) ui_state.zoom_output = zoom;

            ImGui::BeginChild("SyntheticScroll", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);
            ImGui::Image((void*)(intptr_t)textures.syntheticTexture, ImVec2(synthetic_img.cols * zoom, synthetic_img.rows * zoom));
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Synthetic: %d x %d, Zoom: %.1fx", synthetic_img.cols, synthetic_img.rows, zoom);
            ImGui::EndChild();
        }
        ImGui::EndTabItem();
    }

    if (ImGui::BeginTabItem("Focal Template")) {
        cv::Mat template_img = processor.getTemplateImage();
        if (!template_img.empty()) {
            if (ImGui::Button("Save Template Image")) {
                std::string filename = GenerateTimestampedFilename("focal_template", "png");
                if (cv::imwrite(filename, template_img)) {
                    ui_state.last_process_message = "✓ Saved " + filename;
                } else {
                    ui_state.last_process_message = "⚠ Failed to save " + filename;
                }
            }
            ImGui::SameLine();

            ImVec2 available_size = ImGui::GetContentRegionAvail();
            float zoom = ui_state.auto_fit_output ? CalculateFitZoom(template_img, available_size) : ui_state.zoom_output;
            if (ui_state.auto_fit_output) ui_state.zoom_output = zoom;

            ImGui::BeginChild("TemplateScroll", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);
            ImGui::Image((void*)(intptr_t)textures.templateTexture, ImVec2(template_img.cols * zoom, template_img.rows * zoom));
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Template: %d x %d, Zoom: %.1fx\n(This is the last template used for tracking)", template_img.cols, template_img.rows, zoom);
            ImGui::EndChild();
        }
        ImGui::EndTabItem();
    }

    ImGui::EndTabBar();
    ImGui::End();
}

void RenderPlotWindow(SyntheticAperture& processor, UIState& ui_state, std::vector<float>& shiftX, std::vector<float>& shiftY) {
    if (!ui_state.show_plot_window) return;

    static bool first_show = true;
    if (first_show) {
        ImGui::SetNextWindowPos(ImVec2(390, 460));
        ImGui::SetNextWindowSize(ImVec2(300, 400));
        first_show = false;
    }

    ImGui::Begin("Motion Analysis (Template 1)", &ui_state.show_plot_window);

    if (!processor.isProcessed() || shiftX.empty()) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "No motion data available");
        ImGui::End();
        return;
    }

    if (ImPlot::BeginPlot("Camera Motion Path", ImVec2(-1, 250))) {
        ImPlot::SetupAxes("X Displacement", "Y Displacement");
        ImPlot::SetupAxis(ImAxis_Y1, nullptr, ImPlotAxisFlags_Invert);
        ImPlot::PlotLine("Motion Path", shiftX.data(), shiftY.data(), shiftX.size());
        ImPlot::EndPlot();
    }

    if (ImGui::CollapsingHeader("Motion Statistics", ImGuiTreeNodeFlags_DefaultOpen)) {
        float max_x = *std::max_element(shiftX.begin(), shiftX.end());
        float min_x = *std::min_element(shiftX.begin(), shiftX.end());
        float max_y = *std::max_element(shiftY.begin(), shiftY.end());
        float min_y = *std::min_element(shiftY.begin(), shiftY.end());
        ImGui::Text("X Range: %.1f to %.1f (%.1f total)", min_x, max_x, max_x - min_x);
        ImGui::Text("Y Range: %.1f to %.1f (%.1f total)", min_y, max_y, max_y - min_y);
        ImGui::Text("Frames: %d", (int)shiftX.size());
    }

    ImGui::End();
}

void SetupMainMenuBar(UIState& ui_state) {
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("Configuration", nullptr, &ui_state.show_config_window);
            ImGui::MenuItem("Properties", nullptr, &ui_state.show_properties_window);
            ImGui::MenuItem("Input Frame", nullptr, &ui_state.show_input_window);
            ImGui::MenuItem("Output Results", nullptr, &ui_state.show_output_window);
            ImGui::MenuItem("Motion Analysis", nullptr, &ui_state.show_plot_window);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
}

int main(int, char**) {
    if (!glfwInit()) return 1;

    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* window = glfwCreateWindow(1800, 1200, "Synthetic Aperature", nullptr, nullptr);
    if (window == nullptr) return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 5.0f;
    style.FrameRounding = 3.0f;
    style.Colors[ImGuiCol_WindowBg] = ImVec4(0.1f, 0.1f, 0.13f, 1.0f);
    style.Colors[ImGuiCol_MenuBarBg] = ImVec4(0.16f, 0.16f, 0.21f, 1.0f);

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    SyntheticAperture processor;
    SA_Parameters params;
    UIState ui_state;
    TextureManager textures;
    std::vector<float> shiftX, shiftY;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame(); ImGui_ImplGlfw_NewFrame(); ImGui::NewFrame();

        SetupMainMenuBar(ui_state);
        RenderConfigWindow(processor, params, ui_state, textures);
        RenderPropertiesWindow(processor, params, ui_state);
        RenderInputWindow(processor, params, ui_state, textures);
        RenderOutputWindow(processor, ui_state, textures);
        RenderPlotWindow(processor, ui_state, shiftX, shiftY);

        if (textures.needs_update && processor.isProcessed()) {
            MatToTexture(processor.getTemplateImage(), textures.templateTexture);
            MatToTexture(processor.getSyntheticImage(), textures.syntheticTexture);
            MatToTexture(processor.getDepthMap(), textures.depthMapTexture);

            shiftX.clear();
            shiftY.clear();
            const auto& shifts_for_plot = processor.getShifts();
            for (const auto& shift : shifts_for_plot) {
                shiftX.push_back(shift.x);
                shiftY.push_back(shift.y);
            }
            textures.needs_update = false;
        }

        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.12f, 0.14f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

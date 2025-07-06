cmake_minimum_required(VERSION 3.16)
project(SyntheticApertureGUI CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Silence OpenGL deprecation warnings on macOS
if(APPLE)
    add_definitions(-DGL_SILENCE_DEPRECATION)
endif()

find_package(OpenCV REQUIRED)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found!")
endif()

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

add_library(imgui
    vendor/imgui/imgui.cpp
    vendor/imgui/imgui_draw.cpp
    vendor/imgui/imgui_tables.cpp
    vendor/imgui/imgui_widgets.cpp
    vendor/imgui/backends/imgui_impl_glfw.cpp
    vendor/imgui/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC vendor/imgui vendor/imgui/backends)

add_library(implot
    vendor/implot/implot.cpp
    vendor/implot/implot_items.cpp
)
target_include_directories(implot PUBLIC vendor/implot)
target_link_libraries(implot PUBLIC imgui)

add_library(SyntheticApertureLib
    lib/SyntheticAperture.cpp
)
target_link_libraries(SyntheticApertureLib PUBLIC ${OpenCV_LIBS})
target_include_directories(SyntheticApertureLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/lib)


# --- Define the Main Executable ---
add_executable(SyntheticApertureApp src/main.cpp)

target_link_libraries(SyntheticApertureApp
    PRIVATE
    SyntheticApertureLib
    implot
    ${OpenCV_LIBS}
    glfw
    ${OPENGL_LIBRARIES}
)
target_include_directories(SyntheticApertureApp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/lib)

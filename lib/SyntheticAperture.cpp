#include "SyntheticAperture.h"
#include <iostream>

SyntheticAperture::SyntheticAperture()
    : m_video_loaded(false), m_is_processed(false), m_status_message("Ready.") {}

bool SyntheticAperture::loadVideo(const std::string& video_path, const SA_Parameters& params) {
    m_status_message = "Loading video...";
    std::cout << "--- Step 1: Loading and Preparing Video Frames ---" << std::endl;
    m_video_loaded = false;
    m_is_processed = false;
    m_frames_gray.clear();
    m_frames_color.clear();
    m_multi_template_shifts.clear();
    m_parallaxes.clear();
    m_depth_map = cv::Mat();
    m_synthetic_image = cv::Mat();

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        m_status_message = "FATAL ERROR: Video file not found at '" + video_path + "'";
        std::cerr << m_status_message << std::endl;
        return false;
    }

    int frame_count = 0;
    while (cap.isOpened() && frame_count < params.max_frames) {
        cv::Mat frame;
        if (!cap.read(frame)) break;

        if (params.override_width > 0 && params.override_height > 0) {
            cv::resize(frame, frame, cv::Size(params.override_width, params.override_height));
        }

        if (params.rotation != 0) {
            cv::Mat rotated_frame;
            cv::Point2f center((frame.cols - 1) / 2.0, (frame.rows - 1) / 2.0);
            cv::Mat rot = cv::getRotationMatrix2D(center, params.rotation, 1.0);
            cv::warpAffine(frame, rotated_frame, rot, frame.size());
            frame = rotated_frame;
        }

        cv::Mat small_color, small_gray;
        cv::resize(frame, small_color, cv::Size(), 1.0 / params.scale_factor, 1.0 / params.scale_factor);
        cv::cvtColor(small_color, small_gray, cv::COLOR_BGR2GRAY);

        m_frames_gray.push_back(small_gray);
        m_frames_color.push_back(small_color);
        frame_count++;
    }
    cap.release();

    if (m_frames_gray.empty()) {
        m_status_message = "Error: No frames were loaded from the video.";
        std::cerr << m_status_message << std::endl;
        return false;
    }

    m_first_color_frame = m_frames_color[0].clone();
    m_video_loaded = true;
    m_status_message = "Successfully loaded " + std::to_string(m_frames_gray.size()) + " frames.";
    std::cout << m_status_message << "\n" << std::endl;
    return true;
}

bool SyntheticAperture::process(const SA_Parameters& params) {
    if (!m_video_loaded) {
        m_status_message = "Cannot process. Load a video first.";
        std::cerr << m_status_message << std::endl;
        return false;
    }

    m_params = params;
    m_is_processed = false;

    if (m_params.template_points.empty()) {
        m_status_message = "Error: No templates have been selected.";
        std::cerr << m_status_message << std::endl;
        return false;
    }
    cv::Rect frame_rect(0, 0, m_frames_gray[0].cols, m_frames_gray[0].rows);
    for(const auto& pt : m_params.template_points) {
        cv::Rect template_rect(pt.x, pt.y, m_params.template_size, m_params.template_size);
        if ((template_rect & frame_rect) != template_rect) {
            m_status_message = "Error: A template is outside frame boundaries.";
            std::cerr << m_status_message << std::endl;
            return false;
        }
    }

    m_status_message = "Processing... Calculating shifts for all templates.";
    calculateMultiTemplateShifts();

    m_status_message = "Processing... Creating depth map.";
    createDepthMap();

    m_status_message = "Processing... Creating synthetic image (using first template).";
    createSyntheticImage();

    m_is_processed = true;
    m_status_message = "Processing complete!";
    return true;
}

void SyntheticAperture::calculateMultiTemplateShifts() {
    std::cout << "--- Step 2 & 3: Calculating Pixel Shift for Multiple Templates ---" << std::endl;
    m_multi_template_shifts.clear();

    int search_margin = (m_params.search_window_size - m_params.template_size) / 2;

    for (const auto& template_origin : m_params.template_points) {
        cv::Rect template_roi(template_origin.x, template_origin.y, m_params.template_size, m_params.template_size);
        m_template_image = m_frames_gray[0](template_roi);

        cv::Point search_offset(template_origin.x - search_margin, template_origin.y - search_margin);
        std::vector<cv::Point2f> current_template_shifts;

        for (size_t i = 0; i < m_frames_gray.size(); ++i) {
            if (i == 0) {
                current_template_shifts.emplace_back(0, 0);
                continue;
            }

            cv::Rect search_window_roi(search_offset.x, search_offset.y, m_params.search_window_size, m_params.search_window_size);
            search_window_roi &= cv::Rect(0, 0, m_frames_gray[i].cols, m_frames_gray[i].rows);

            cv::Mat search_window = m_frames_gray[i](search_window_roi);
            cv::Mat correlation_map;
            cv::matchTemplate(search_window, m_template_image, correlation_map, cv::TM_CCOEFF_NORMED);

            cv::Point peak_loc;
            cv::minMaxLoc(correlation_map, nullptr, nullptr, nullptr, &peak_loc);

            float sx = (search_window_roi.x + peak_loc.x) - template_origin.x;
            float sy = (search_window_roi.y + peak_loc.y) - template_origin.y;

            current_template_shifts.emplace_back(sx, sy);
        }
        m_multi_template_shifts.push_back(current_template_shifts);
    }
    std::cout << "Finished calculating all pixel shifts for " << m_multi_template_shifts.size() << " templates.\n" << std::endl;
}

void SyntheticAperture::createDepthMap() {
    std::cout << "--- Step 4: Creating Depth Map ---" << std::endl;
    m_parallaxes.clear();
    m_depth_map = cv::Mat::zeros(m_first_color_frame.size(), CV_8UC3);

    if (m_multi_template_shifts.size() < 2) {
        m_status_message = "Depth map requires at least 2 templates.";
        std::cout << m_status_message << std::endl;
        m_depth_map = m_first_color_frame.clone();
        cv::putText(m_depth_map, m_status_message, cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,255), 2);
        return;
    }


    float min_parallax = std::numeric_limits<float>::max();
    float max_parallax = std::numeric_limits<float>::min();

    for (const auto& shifts : m_multi_template_shifts) {
        if (shifts.empty()) {
            m_parallaxes.push_back(0);
            continue;
        }
        float parallax = cv::norm(shifts.back());
        m_parallaxes.push_back(parallax);
        min_parallax = std::min(min_parallax, parallax);
        max_parallax = std::max(max_parallax, parallax);
    }

    float parallax_range = max_parallax - min_parallax;

    for (size_t i = 0; i < m_parallaxes.size(); ++i) {
        float normalized_p = 0.0f;
        if (parallax_range > 1e-5) { // Avoid division by zero
            normalized_p = (m_parallaxes[i] - min_parallax) / parallax_range;
        }

        // Color: Blue (far, min parallax) to Red (near, max parallax)
        cv::Scalar color(255 * (1.0 - normalized_p), 0, 255 * normalized_p);

        cv::Point center = m_params.template_points[i] + cv::Point(m_params.template_size / 2, m_params.template_size / 2);
        int radius = m_params.template_size;
        cv::circle(m_depth_map, center, radius, color, -1, cv::LINE_AA);
    }

    std::cout << "Depth map created successfully.\n" << std::endl;
}

void SyntheticAperture::createSyntheticImage() {
    std::cout << "--- Step 5: Creating Synthetic Aperture Photograph ---" << std::endl;
    if (m_multi_template_shifts.empty()) {
        m_synthetic_image = cv::Mat();
        return;
    }

    const auto& shifts = m_multi_template_shifts[0];

    cv::Mat synthetic_image_float = cv::Mat::zeros(m_frames_color[0].size(), CV_32FC3);

    for (size_t i = 0; i < m_frames_color.size(); ++i) {
        const auto& color_frame = m_frames_color[i];
        float sx = shifts[i].x;
        float sy = shifts[i].y;

        cv::Mat translation_matrix = (cv::Mat_<double>(2, 3) << 1, 0, -sx, 0, 1, -sy);

        cv::Mat shifted_frame;
        cv::warpAffine(color_frame, shifted_frame, translation_matrix, color_frame.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));

        cv::Mat shifted_float;
        shifted_frame.convertTo(shifted_float, CV_32FC3);

        synthetic_image_float += shifted_float;
    }

    synthetic_image_float /= (float)m_frames_color.size();
    synthetic_image_float.convertTo(m_synthetic_image, CV_8UC3);
    std::cout << "Synthetic aperture photograph created successfully.\n" << std::endl;
}


const cv::Mat& SyntheticAperture::getFirstColorFrame() const { return m_first_color_frame; }
const cv::Mat& SyntheticAperture::getTemplateImage() const { return m_template_image; }
const cv::Mat& SyntheticAperture::getSyntheticImage() const { return m_synthetic_image; }
const std::string& SyntheticAperture::getStatusMessage() const { return m_status_message; }
bool SyntheticAperture::isVideoLoaded() const { return m_video_loaded; }
bool SyntheticAperture::isProcessed() const { return m_is_processed; }


const cv::Mat& SyntheticAperture::getDepthMap() const {
    return m_depth_map;
}

const std::vector<cv::Point2f>& SyntheticAperture::getShifts() const {
    static const std::vector<cv::Point2f> empty_shifts;
    return m_multi_template_shifts.empty() ? empty_shifts : m_multi_template_shifts[0];
}

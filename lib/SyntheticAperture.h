#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

//Config params
struct SA_Parameters {
    int max_frames = 90;
    int scale_factor = 2;
    std::vector<cv::Point> template_points;
    int template_size = 32;
    int search_window_size = 160;
    int override_width = 0;
    int override_height = 0;
    int rotation = 0;
};

class SyntheticAperture {
public:
    SyntheticAperture();

    bool loadVideo(const std::string& video_path, const SA_Parameters& params);
    bool process(const SA_Parameters& params);

    const cv::Mat& getFirstColorFrame() const;
    const cv::Mat& getTemplateImage() const;
    const cv::Mat& getSyntheticImage() const;
    const cv::Mat& getDepthMap() const;
    const std::vector<cv::Point2f>& getShifts() const;
    const std::string& getStatusMessage() const;
    bool isVideoLoaded() const;
    bool isProcessed() const;

private:
    void calculateMultiTemplateShifts();
    void createDepthMap();
    void createSyntheticImage();

    SA_Parameters m_params;
    std::string m_status_message;

    std::vector<cv::Mat> m_frames_gray;
    std::vector<cv::Mat> m_frames_color;

    cv::Mat m_first_color_frame;
    cv::Mat m_template_image;
    cv::Mat m_synthetic_image;

    cv::Mat m_depth_map;
    std::vector<float> m_parallaxes;
    std::vector<std::vector<cv::Point2f>> m_multi_template_shifts;

    bool m_video_loaded;
    bool m_is_processed;
};

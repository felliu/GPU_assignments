#include "export.h"

#include <opencv2/opencv.hpp>

void export_image(int height, int width, float* pixels) {
    std::vector<cv::Scalar> pixels_cv;
    cv::Mat img_mat_f(height, width, CV_32FC3, static_cast<void*>(pixels));
    cv::Mat img;

    img_mat_f.convertTo(img, CV_8UC3, 255.0f);

    cv::imwrite(std::string("scene.png"), img);
}

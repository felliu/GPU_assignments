#include "export.h"

#include <opencv2/opencv.hpp>

void export_image(int height, int width, float* pixels) {
    cv::Mat img_mat_f(height, width, CV_32F, static_cast<void*>(pixels));
    cv::Mat img;

    img_mat_f *= 255.0f;

    img_mat_f.convertTo(img, CV_8UC3);

    cv::imwrite(std::string("scene.png"), img);
}

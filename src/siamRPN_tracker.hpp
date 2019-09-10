#ifndef _SIAMRPN_TRACKER_HPP_
#define _SIAMRPN_TRACKER_HPP_

#include <cmath>
#include <iostream>
#include <vector>

#include <ATen/Tensor.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <opencv2/opencv.hpp>

#define ANCH_NUM 5

struct Anchor {
    float x;
    float y;
    float w;
    float h;
};

struct Parameter {
    int exemplar_sz = 127;
    int instance_sz = 271;
    int total_stride = 8;
    float penalty_k = 0.055;
    float win_influence = 0.42;
    float lr = 0.295;
};

template <typename t>
t x2(const cv::Rect_<t>& rect) {
    return rect.x + rect.width;
}

template <typename t>
t y2(const cv::Rect_<t>& rect) {
    return rect.y + rect.height;
}

template <typename t>
void limit(cv::Rect_<t>& rect, cv::Rect_<t> limit) {
    if (rect.x + rect.width > limit.x + limit.width) {
        rect.width = limit.x + limit.width - rect.x;
    }
    if (rect.y + rect.height > limit.y + limit.height) {
        rect.height = limit.y + limit.height - rect.y;
    }
    if (rect.x < limit.x) {
        rect.width -= (limit.x - rect.x);
        rect.x = limit.x;
    }
    if (rect.y < limit.y) {
        rect.height -= (limit.y - rect.y);
        rect.y = limit.y;
    }
    if (rect.width < 0) {
        rect.width = 0;
    }
    if (rect.height < 0) {
        rect.height = 0;
    }
}

template <typename t>
void limit(cv::Rect_<t>& rect, t width, t height, t x = 0, t y = 0) {
    limit(rect, cv::Rect_<t>(x, y, width, height));
}

template <typename t>
cv::Rect getBorder(const cv::Rect_<t>& original, cv::Rect_<t>& limited) {
    cv::Rect_<t> res;
    res.x = limited.x - original.x;
    res.y = limited.y - original.y;
    res.width = x2(original) - x2(limited);
    res.height = y2(original) - y2(limited);
    assert(res.x >= 0 && res.y >= 0 && res.width >= 0 && res.height >= 0);
    return res;
}

cv::Mat subwindow(const cv::Mat& in, const cv::Rect& window, int borderType = cv::BORDER_CONSTANT);

class SiamRPNTracker {
   public:
    SiamRPNTracker(const std::shared_ptr<torch::jit::script::Module>& temple_net,
                   const std::shared_ptr<torch::jit::script::Module>& detect_net);

    void init(const cv::Mat& img, cv::Rect2d& box);

    void update(const cv::Mat& img, cv::Rect2d& box);

   private:
    void createAnchors(const int& response_sz);

    std::vector<float> createPenalty(const float& target_w, const float& target_h, const std::vector<Anchor>& offsets);

    void calculateHann(const cv::Size& sz, cv::Mat& output);

    cv::Mat getSamplePatch(const cv::Mat im, const cv::Point2f posf, const int& in_sz, const int& out_sz);

    float z_sz_;
    float x_sz_;
    std::vector<Anchor> anchors_;
    Parameter cfg_;
    cv::Mat hann_window_;

    cv::Point2f pos_;
    float target_sz_w_;
    float target_sz_h_;
    float base_target_sz_w_;
    float base_target_sz_h_;
    at::Tensor kernel_reg_;
    at::Tensor kernel_cls_;

    std::shared_ptr<torch::jit::script::Module> temple_net_;
    std::shared_ptr<torch::jit::script::Module> detect_net_;
};

#endif
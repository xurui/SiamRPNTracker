#include <fstream>
#include <iostream>
#include <string>
#include "siamRPN_tracker.hpp"

using namespace std;
using namespace cv;

cv::Rect g_rectangle;
bool g_bDrawingBox = false;

void readGroundTruthFromFile(ifstream* groundtruth, Rect2d& bboxGroundtruth) {
    float x, y, w, h;
    std::string s;
    getline(*groundtruth, s, ',');
    x = atof(s.c_str());
    getline(*groundtruth, s, ',');
    y = atof(s.c_str());
    getline(*groundtruth, s, ',');
    w = atof(s.c_str());
    getline(*groundtruth, s);
    h = atof(s.c_str());
    cout << "Bounding box:" << x << " " << y << " " << w << " " << h << " " << endl;
    bboxGroundtruth.x = x;
    bboxGroundtruth.y = y;
    bboxGroundtruth.width = w;
    bboxGroundtruth.height = h;
}

int main(int argc, char** argv) {
    // step 1. Read the groundtruth and image
    // step 1.1 Read the groundtruth bbox
    std::string path = "../sequences/Lemming";
    std::string gt_file_name = path + "/groundtruth_rect.txt";

    // Read the groundtruth bbox
    ifstream* groundtruth;
    groundtruth = new ifstream(path + "/groundtruth_rect.txt");
    Rect2d bboxGroundtruth;
    readGroundTruthFromFile(groundtruth, bboxGroundtruth);
    Rect2d bbox = bboxGroundtruth;

    int f = 1;
    std::string img_file_name = path + "/img/" + cv::format("%04d", f) + ".jpg";
    cv::Mat frame = cv::imread(img_file_name);
    cv::Mat frameDraw;
    frame.copyTo(frameDraw);
    if (!frame.data) {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    rectangle(frameDraw, bboxGroundtruth, Scalar(0, 0, 0), 2, 1);

    // step 2. tracking process
    // step 2.1 tracking init process
    std::shared_ptr<torch::jit::script::Module> temple_net = torch::jit::load("../model/template.pt");
    temple_net->to(at::kCUDA);

    std::shared_ptr<torch::jit::script::Module> detect_net = torch::jit::load("../model/detection.pt");
    detect_net->to(at::kCUDA);

    SiamRPNTracker siamRPN_tracker(temple_net, detect_net);
    siamRPN_tracker.init(frame, bbox);
    int cnt = 0;
    // step 2.2 tracking update process
    while (frame.data) {
        frame.copyTo(frameDraw);
        siamRPN_tracker.update(frame, bbox);
        cnt++;
        rectangle(frameDraw, bbox, Scalar(255, 0, 255), 2, 1);  // blue
        rectangle(frameDraw, bboxGroundtruth, Scalar(0, 0, 0), 2, 1);
        imshow("siamRPNTracker", frameDraw);

        std::cout << "--bboxGroundtruth x: " << bboxGroundtruth.x << " y:" << bboxGroundtruth.y
                  << " width: " << bboxGroundtruth.width << " height: " << bboxGroundtruth.height << std::endl;
        std::cout << "--bbox x: " << bbox.x << " y:" << bbox.y << " width: " << bbox.width << " height: " << bbox.height
                  << std::endl;
        int k = cv::waitKey(1);
        // press 'esc' or 'q'
        if (k == 27 || k == 'q' || k == 'Q') {
            break;
        } else if (k == 32) {
            cv::waitKey();
        }
        f++;
        readGroundTruthFromFile(groundtruth, bboxGroundtruth);
        img_file_name = path + "/img/" + cv::format("%04d", f) + ".jpg";
        frame = cv::imread(img_file_name);
        if (!frame.data) {
            break;
        }
    }

    return 0;
}
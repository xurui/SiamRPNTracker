[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)
[![MIT Licence](https://badges.frapsoft.com/os/mit/mit.svg?v=103)](https://opensource.org/licenses/mit-license.php)

# siamrpn_tracker

A Libtorch implementation of the SiamRPN tracker algorithm described in paper High Performance Visual Tracking with Siamese Region Proposal Network. This project is inspired by the pytorch version, I rewritten it with C++.

![image](https://github.com/xurui/SiamRPNTracker/blob/master/model/siamrpn.png )

# Requirements
1. LibTorch v1.0.0
2. CUDA
3. OpenCV

# To Compile

If you want to compile and run the project, you can create a build folder first, and set CMAKE_PREFIX_PATH as /Path/to/pytorch/torch/share/cmake, then run the command:
```
mkdir build;
cd build;
cmake ..;
make;
```

# Quick Start
1. Download traced model files detect.pt and template.pt from [Baidu Yun](https://pan.baidu.com/s/1KRPXsHMfxkdzlxrru11-Mg) [Google Drive](https://drive.google.com/drive/folders/1DxQPgbjk5V7bMG0NOrH5dhqfVd03CE1o?usp=sharing)
, and put the file under path/to/SiamRPNTracker/model.
2. run 
```
./siamrpn_tracker
```

# Reference

Li, Bo, et al. "High performance visual tracking with siamese region proposal network." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.

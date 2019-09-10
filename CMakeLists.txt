cmake_minimum_required(VERSION 2.8.3)
project(siamrpn_tracker)

set(CMAKE_PREFIX_PATH /Path/to/pytorch/torch/share/cmake)

find_package(Torch REQUIRED)
find_package( OpenCV REQUIRED )

include_directories( ./ ${OpenCV_INCLUDE_DIRS} )

add_executable(siamrpn_tracker src/run_tracker.cpp src/siamRPN_tracker.cpp)
target_link_libraries(siamrpn_tracker "${TORCH_LIBRARIES}"  ${OpenCV_LIBS})
set_property(TARGET siamrpn_tracker PROPERTY CXX_STANDARD 11)

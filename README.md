# camera_calibration
This is a monocular camera calibration project without using cv::calibrateCamera() function. The calibration is based on Zhang<a href="#1">[1]</a>. The project uses OpenCV to read images and find chessboards, uses Eigen to linearly solve initial parameters and uses Ceres for nonlinear optimization. During optimization, you can choose to use automatic differentiation or analytical differentiation.

## Parameters to be solved
* Camera intrinsics : alpha, beta, u0, v0
* Distortion coefficients : k1, k2, p1, p2
* Camera extrinsics : R, t for each image

## Requirements
* Cmake
* OpenCV3
* Eigen3
* Ceres

## Usage
```
cd camera_calibration
mkdir build
cd build/
cmake ..
make
./main <config_file> <is_auto_diff>
```
* <config_file> : string --- Config file in .yaml format.
* <is_auto_diff> : 0/1 --- Whether automatic differentiation is used in nonlinear optimization.

## Citations
<a name="1">[1]</a>
Zhang, Z. (2000). A flexible new technique for camera calibration. IEEE Transactions on pattern analysis and machine intelligence, 22(11), 1330-1334.
#ifndef _CAMERA_CALIBRATION_UTIL_H_
#define _CAMERA_CALIBRATION_UTIL_H_

#include <dirent.h>
#include <sys/stat.h>
#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>


int ReadFilenames(std::vector<std::string>& filenames, const std::string& directory);

double RadianToDegree(double radian);
double DegreeToRadian(double degree);

#endif
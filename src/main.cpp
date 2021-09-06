#include <iostream>

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
// #include <opencv2/core/eigen.hpp>

#include "calibration.h"
#include "util.h"

using namespace std;

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cout << "Usage :" << std::endl
                  << "./main <config_file>" << std::endl;
        return -1;
    }

    std::string config_file = argv[1];
    std::cout << "config_file : " << config_file << endl;
    cv::FileStorage fs(config_file, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cout << "Failed to open " << config_file << " file!" << std::endl;
        return -1;
    }

    std::string image_path;
    Config config;
    fs["image_path"] >> image_path;
    fs["image_size"] >> config.image_size;
    fs["board_size"] >> config.board_size;
    fs["square_size"] >> config.square_size;
    fs.release();

    std::vector<std::string> image_names;
    int num_image = ReadFilenames(image_names, image_path);
    std::cout << "Number of images = " << num_image << std::endl;

    std::vector<cv::Mat> images;
    std::vector<std::vector<cv::Point2f>> pixel_points;
    std::vector<std::vector<cv::Point3f>> object_points;

    int board_count = ReadImageAndFindChessboard(image_path, image_names, config,
        images, pixel_points, object_points);
    std::cout << "Found " << board_count << " chessboards." << std::endl;

    // Calculate Homography for each image.
    std::vector<Eigen::Matrix3d> Hs;
    CalculateHomography(pixel_points, object_points, config, Hs);

    // Calculate intrinsic matrix A
    Eigen::Matrix3d camera_matrix;
    CalculateIntrinsics(Hs, config, camera_matrix);
    std::cout << "camera_matrix:" << std::endl << camera_matrix << std::endl;

    // Calculate extrinsic matrix for each image.
    std::vector<Eigen::Matrix3d> Rs;
    std::vector<Eigen::Vector3d> ts;
    CalculateExtrinsics(Hs, camera_matrix, config, Rs, ts);
    for (int i = 0; i < board_count; i++)
    {
        std::cout << "i = " << i << std::endl
                  << "R:" << std::endl << Rs[i] << std::endl
                  << "t:" << std::endl << ts[i] << std::endl;
    }


    // Calculate distortion coefficients with k1, k2
    Eigen::VectorXd dist_coeffs;
    CalculateDistCoeffs(pixel_points, object_points, 
        camera_matrix, Rs, ts, config, dist_coeffs);
    std::cout << "dist_coeffs = " << std::endl << dist_coeffs << std::endl;

    return 0;
}
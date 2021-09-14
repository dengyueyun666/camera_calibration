#include <iostream>

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <glog/logging.h>
#include <ceres/ceres.h>

#include "calibration.h"
#include "reprojection_error.h"
#include "reprojection_factor.h"
#include "util.h"

using namespace std;

int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
    if (argc != 3)
    {
        std::cout << "Usage :" << std::endl
                  << "./main <config_file> <is_auto_diff>" << std::endl
                  << "<config_file> : string --- Config file in .yaml format." << std::endl
                  << "<is_auto_diff> : 0/1 --- Whether automatic differentiation"
                  << " is used in nonlinear optimization." << std::endl;
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

    bool is_auto_diff = std::atoi(argv[2]) == 1 ? true : false;

    std::vector<std::string> image_names;
    int num_image = ReadFilenames(image_names, image_path);
    std::cout << "Number of image files: " << num_image << std::endl;
    if (num_image == 0)
    {
        std::cerr << "Failed to find image files in " << image_path << std::endl;
        return -1;
    }

    std::vector<cv::Mat> images;
    std::vector<std::vector<cv::Point2f>> pixel_points;
    std::vector<std::vector<cv::Point3f>> object_points;

    int board_count = ReadImageAndFindChessboard(image_path, image_names, config,
        images, pixel_points, object_points);
    std::cout << "Found " << board_count << " chessboards." << std::endl;

    // Calculate Homography for each image.
    std::vector<Eigen::Matrix3d> Hs;
    CalculateHomography(pixel_points, object_points, config, Hs);

    // Calculate intrinsic matrix (alpha, beta, u0, v0)
    Eigen::Matrix3d camera_matrix;
    CalculateIntrinsics(Hs, config, camera_matrix);
    std::cout << "init_cam_mat:" << std::endl << camera_matrix << std::endl;

    // Calculate extrinsic matrix for each image.
    std::vector<Eigen::Matrix3d> Rs;
    std::vector<Eigen::Vector3d> ts;
    CalculateExtrinsics(Hs, camera_matrix, config, Rs, ts);

    // Calculate distortion coefficients (k1, k2, p1, p2)
    Eigen::VectorXd dist_coeffs;
    CalculateDistCoeffs(pixel_points, object_points, 
        camera_matrix, Rs, ts, config, dist_coeffs);
    std::cout << "init_dist_coeffs:" << std::endl << dist_coeffs << std::endl;

    double error = CalculateReprojectionError(pixel_points, object_points,
        camera_matrix, dist_coeffs, Rs, ts, config);
    std::cout << "error(before LM) = " << error << std::endl;

    double cam_mat[4] = {camera_matrix(0, 0), camera_matrix(1, 1),
                         camera_matrix(0, 2), camera_matrix(1, 2)};
    double dist_coes[4] = {dist_coeffs(0), dist_coeffs(1), 
                           dist_coeffs(2), dist_coeffs(3)};
    // double rvecs[27][3];
    // double tvecs[27][3];
    std::vector<std::vector<double>> rvecs(board_count, std::vector<double>(3));
    std::vector<std::vector<double>> tvecs(board_count, std::vector<double>(3));
    for (int i = 0; i < board_count; i++)
    {
        cv::Mat R, rvec;
        cv::eigen2cv(Rs[i], R);
        cv::Rodrigues(R, rvec);
        rvecs[i][0] = rvec.at<double>(0);
        rvecs[i][1] = rvec.at<double>(1);
        rvecs[i][2] = rvec.at<double>(2);
        
        tvecs[i][0] = ts[i](0);
        tvecs[i][1] = ts[i](1);
        tvecs[i][2] = ts[i](2);
    }

    ceres::Problem problem;
    ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
    for (int i = 0; i < board_count; i++)
    {
        const std::vector<cv::Point2f>& pixel_point = pixel_points[i];
        const std::vector<cv::Point3f>& object_point = object_points[i];

        for (int j = 0; j < config.board_size.area(); j++)
        {
            if (is_auto_diff)
            {
                problem.AddResidualBlock(
                    ReprojectionError::Create(pixel_point[j],
                                            object_point[j]),
                    nullptr,
                    &rvecs[i][0], &tvecs[i][0], cam_mat, dist_coes);
            }
            else
            {
                ReprojectionFactor *f = new ReprojectionFactor(pixel_point[j], 
                                                               object_point[j]);
                problem.AddResidualBlock(f, nullptr,
                    cam_mat, dist_coes, &rvecs[i][0], &tvecs[i][0]);
            }
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 300;
    options.minimizer_progress_to_stdout = true;
    options.use_nonmonotonic_steps = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;

    camera_matrix(0, 0) = cam_mat[0];
    camera_matrix(1, 1) = cam_mat[1];
    camera_matrix(0, 2) = cam_mat[2];
    camera_matrix(1, 2) = cam_mat[3];
    std::cout << "refined_cam_mat:" << std::endl << camera_matrix << std::endl;

    dist_coeffs(0) = dist_coes[0];
    dist_coeffs(1) = dist_coes[1];
    dist_coeffs(2) = dist_coes[2];
    dist_coeffs(3) = dist_coes[3];
    std::cout << "refined_dist_coeffs:" << std::endl << dist_coeffs << std::endl;

    for (int i = 0; i < board_count; i++)
    {
        // std::cout << "------------" << std::endl;
        // std::cout << "i = " << i << std::endl;
        // std::cout << "R(before):" << std::endl << Rs[i] << std::endl;
        cv::Mat rvec = (cv::Mat_<double>(3, 1) << rvecs[i][0],
                                                  rvecs[i][1],
                                                  rvecs[i][2]);
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        cv::cv2eigen(R, Rs[i]);
        // std::cout << "R(after):" << std::endl << Rs[i] << std::endl;

        // std::cout << "t(before):" << std::endl << ts[i] << std::endl;
        ts[i](0) = tvecs[i][0];
        ts[i](1) = tvecs[i][1];
        ts[i](2) = tvecs[i][2];
        // std::cout << "t(after):" << std::endl << ts[i] << std::endl;
    }

    error = CalculateReprojectionError(pixel_points, object_points,
        camera_matrix, dist_coeffs, Rs, ts, config);
    std::cout << "error(after LM) = " << error << std::endl;

    cv::Mat refined_cam_mat = cv::Mat::eye(3, 3, CV_64F);
    refined_cam_mat.at<double>(0, 0) = cam_mat[0];
    refined_cam_mat.at<double>(1, 1) = cam_mat[1];
    refined_cam_mat.at<double>(0, 2) = cam_mat[2];
    refined_cam_mat.at<double>(1, 2) = cam_mat[3];

    cv::Mat refined_dist_coes = cv::Mat::zeros(4, 1, CV_64F);
    refined_dist_coes.at<double>(0) = dist_coes[0];
    refined_dist_coes.at<double>(1) = dist_coes[1];
    refined_dist_coes.at<double>(2) = dist_coes[2];
    refined_dist_coes.at<double>(3) = dist_coes[3];
    
    cv::Mat map1, map2;
    cv::initUndistortRectifyMap(refined_cam_mat, refined_dist_coes,
        cv::Mat(), refined_cam_mat, config.image_size,
        CV_16SC2, map1, map2);

    for (int i = 0; i < board_count; i++)
    {
        cv::Mat undistorted_img;
        cv::remap(images[i], undistorted_img, map1, map2, cv::INTER_LINEAR,
            cv::BORDER_CONSTANT, cv::Scalar());
        cv::imshow("Original", images[i]);
        cv::imshow("Undistorted", undistorted_img);
        if ((cv::waitKey(0) & 255) == 27)
            break;
    }

    return 0;
}
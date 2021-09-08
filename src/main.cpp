#include <iostream>

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <glog/logging.h>
#include <ceres/ceres.h>

#include "calibration.h"
#include "reprojection_error.h"
#include "util.h"

using namespace std;

int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
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

    // Calculate intrinsic matrix A
    Eigen::Matrix3d camera_matrix;
    CalculateIntrinsics(Hs, config, camera_matrix);
    std::cout << "camera_matrix:" << std::endl << camera_matrix << std::endl;

    // Calculate extrinsic matrix for each image.
    std::vector<Eigen::Matrix3d> Rs;
    std::vector<Eigen::Vector3d> ts;
    CalculateExtrinsics(Hs, camera_matrix, config, Rs, ts);
    // for (int i = 0; i < board_count; i++)
    // {
    //     std::cout << "i = " << i << std::endl
    //               << "R:" << std::endl << Rs[i] << std::endl
    //               << "t:" << std::endl << ts[i] << std::endl;
    // }

    // Calculate distortion coefficients with k1, k2
    Eigen::VectorXd dist_coeffs;
    CalculateDistCoeffs(pixel_points, object_points, 
        camera_matrix, Rs, ts, config, dist_coeffs);
    std::cout << "dist_coeffs = " << std::endl << dist_coeffs << std::endl;

    double error = CalculateReprojectionError(pixel_points, object_points,
        camera_matrix, dist_coeffs, Rs, ts, config);
    std::cout << "error(before) = " << std::endl << error << std::endl;

    double cam_mat[5] = {camera_matrix(0, 0), camera_matrix(1, 1), 
                         camera_matrix(0, 1),
                         camera_matrix(0, 2), camera_matrix(1, 2)};
    double dist_coes[4] = {dist_coeffs(0), dist_coeffs(1), 
                           dist_coeffs(2), dist_coeffs(3)};
    double rvecs[27][3];
    double tvecs[27][3];
    double qvecs[27][4];
    for (int i = 0; i < board_count; i++)
    {
        // std::cout << "i = " << i << std::endl;

        cv::Mat R, rvec;
        cv::eigen2cv(Rs[i], R);
        cv::Rodrigues(R, rvec);
        rvecs[i][0] = rvec.at<double>(0);
        rvecs[i][1] = rvec.at<double>(1);
        rvecs[i][2] = rvec.at<double>(2);
        // std::cout << "CVR:" << std::endl << rvec << std::endl;
        // std::cout << "rvecs[i]:" << std::endl;
        // std::cout << rvecs[i][0] << std::endl;
        // std::cout << rvecs[i][1] << std::endl;
        // std::cout << rvecs[i][2] << std::endl;

        // Eigen::AngleAxisd R_angle(Rs[i]);
        // Eigen::Vector3d R_vec = R_angle.angle() * R_angle.axis();
        // rvecs[i][0] = R_vec(0);
        // rvecs[i][1] = R_vec(1);
        // rvecs[i][2] = R_vec(2);
        // std::cout << "EigenR:" << std::endl << R_vec << std::endl;
        // std::cout << "rvecs[i]:" << std::endl;
        // std::cout << rvecs[i][0] << std::endl;
        // std::cout << rvecs[i][1] << std::endl;
        // std::cout << rvecs[i][2] << std::endl;

        // Eigen::Quaterniond q(Rs[i]);
        // qvecs[i][0] = q.x();
        // qvecs[i][1] = q.y();
        // qvecs[i][2] = q.z();
        // qvecs[i][3] = q.w();
        // std::cout << "q:" << std::endl << q << std::endl;
        
        tvecs[i][0] = ts[i](0);
        tvecs[i][1] = ts[i](1);
        tvecs[i][2] = ts[i](2);
    }

    ceres::Problem problem;
    ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
    // ceres::LocalParameterization* q_para = new ceres::QuaternionParameterization();
    for (int i = 0; i < board_count; i++)
    {
        const std::vector<cv::Point2f>& pixel_point = pixel_points[i];
        const std::vector<cv::Point3f>& object_point = object_points[i];

        for (int j = 0; j < config.board_size.area(); j++)
        {
            problem.AddResidualBlock(
                ReprojectionError::Create(pixel_point[j],
                                          object_point[j]),
                nullptr,
                rvecs[i], tvecs[i], cam_mat, dist_coes);
        }
        // problem.SetParameterization(qvecs[i], q_para);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.trust_region_strategy_type = ceres::DOGLEG;
    // options.max_num_iterations = 30;
    options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;

    camera_matrix(0, 0) = cam_mat[0];
    camera_matrix(1, 1) = cam_mat[1];
    camera_matrix(0, 1) = cam_mat[2];
    camera_matrix(0, 2) = cam_mat[3];
    camera_matrix(1, 2) = cam_mat[4];
    std::cout << "refined_camera_matrix:" << std::endl << camera_matrix << std::endl;

    dist_coeffs(0) = dist_coes[0];
    dist_coeffs(1) = dist_coes[1];
    dist_coeffs(2) = dist_coes[2];
    dist_coeffs(3) = dist_coes[3];
    std::cout << "refined_dist_coeffs:" << std::endl << dist_coeffs << std::endl;

    for (int i = 0; i < board_count; i++)
    {
        std::cout << "------------" << std::endl;
        std::cout << "i = " << i << std::endl;
        std::cout << "R(before):" << std::endl << Rs[i] << std::endl;
        cv::Mat rvec = (cv::Mat_<double>(3, 1) << rvecs[i][0],
                                                  rvecs[i][1],
                                                  rvecs[i][2]);
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        cv::cv2eigen(R, Rs[i]);
        std::cout << "R(after):" << std::endl << Rs[i] << std::endl;

        // std::cout << "R(before):" << std::endl << Rs[i] << std::endl;
        // Eigen::Quaterniond q(qvecs[i][3], qvecs[i][0], qvecs[i][1], qvecs[i][2]);
        // Rs[i] = q.toRotationMatrix();
        // std::cout << "R(after):" << std::endl << Rs[i] << std::endl;

        std::cout << "t(before):" << std::endl << ts[i] << std::endl;
        ts[i](0) = tvecs[i][0];
        ts[i](1) = tvecs[i][1];
        ts[i](2) = tvecs[i][2];
        std::cout << "t(after):" << std::endl << ts[i] << std::endl;
    }

    error = CalculateReprojectionError(pixel_points, object_points,
        camera_matrix, dist_coeffs, Rs, ts, config);
    std::cout << "error(after) = " << std::endl << error << std::endl;

    cv::Mat refined_cam_mat = cv::Mat::eye(3, 3, CV_64F);
    refined_cam_mat.at<double>(0, 0) = cam_mat[0];
    refined_cam_mat.at<double>(1, 1) = cam_mat[1];
    refined_cam_mat.at<double>(0, 1) = cam_mat[2];
    refined_cam_mat.at<double>(0, 2) = cam_mat[3];
    refined_cam_mat.at<double>(1, 2) = cam_mat[4];
    std::cout << "refined_cam_mat:" << std::endl << refined_cam_mat << std::endl;

    cv::Mat refined_dist_coes = cv::Mat::zeros(4, 1, CV_64F);
    refined_dist_coes.at<double>(0) = dist_coes[0];
    refined_dist_coes.at<double>(1) = dist_coes[1];
    refined_dist_coes.at<double>(2) = dist_coes[2];
    refined_dist_coes.at<double>(3) = dist_coes[3];
    std::cout << "refined_dist_coes:" << std::endl << refined_dist_coes << std::endl;
    
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
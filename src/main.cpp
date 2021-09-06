#include <iostream>

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
// #include <opencv2/core/eigen.hpp>

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

    string config_file = argv[1];
    std::cout << "config_file : " << config_file << endl;
    cv::FileStorage fs(config_file, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cout << "Failed to open " << config_file << " file!" << std::endl;
        return -1;
    }

    string image_path;
    cv::Size image_size;
    cv::Size board_size;
    float square_size;
    fs["image_path"] >> image_path;
    fs["image_size"] >> image_size;
    fs["board_size"] >> board_size;
    fs["square_size"] >> square_size;
    fs.release();

    vector<string> image_names;
    int num_image = ReadFilenames(image_names, image_path);
    std::cout << "Number of images = " << num_image << std::endl;

    vector<vector<cv::Point2f>> image_points;
    vector<vector<cv::Point3f>> object_points;

    int board_count = 0;
    for (size_t i = 0; i < image_names.size(); i++)
    {
        cv::Mat img = cv::imread(image_path + image_names[i]);
        if (img.empty())
        {
            std::cerr << "Failed to read " << image_names[i] 
                      << " image!" << std::endl;
            continue;
        }
        
        vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(img, board_size, corners);

        if (!found)
        {
            std::cerr << "Failed to find chessboard in " << image_names[i] 
                      << " image!" << std::endl;
            continue;
        }
        else
        {
            board_count++;
            image_points.push_back(corners);
            object_points.push_back(vector<cv::Point3f>());
            std::vector<cv::Point3f> &obj_pts = object_points.back();
            int board_area = board_size.area();
            obj_pts.resize(board_area);
            for (int j = 0; j < board_area; j++)
            {
                obj_pts[j] = cv::Point3f(static_cast<float>(j / board_size.width),
                                         static_cast<float>(j % board_size.width),
                                         0.0f) * square_size;
            }

            // cv::drawChessboardCorners(img, board_size, corners, found);
            // cv::imshow("Calibration", img);
            // cv::waitKey();
        }
    }
    std::cout << "Found " << board_count << " chessboards." << std::endl;

    // Solve Homography for each image.
    std::vector<Eigen::Matrix3d> Hs(board_count);
    for (int i = 0; i < board_count; i++)
    {
        int board_area = board_size.area();
        const vector<cv::Point2f>& image_point = image_points[i];
        const vector<cv::Point3f>& object_point = object_points[i];

        cv::Point2d mean_img_pt(0, 0);
        cv::Point2d mean_obj_pt(0, 0);
        for (int j = 0; j < board_area; j++)
        {
            mean_img_pt.x += image_point[j].x;
            mean_img_pt.y += image_point[j].y;

            mean_obj_pt.x += object_point[j].x;
            mean_obj_pt.y += object_point[j].y;
        }
        mean_img_pt /= board_area;
        mean_obj_pt /= board_area;

        double mean_img_pt_dist = 0;
        double mean_obj_pt_dist = 0;
        for (int j = 0; j < board_area; j++)
        {
            mean_img_pt_dist += std::sqrt(
                std::pow(image_point[j].x - mean_img_pt.x, 2) +
                std::pow(image_point[j].y - mean_img_pt.y, 2));

            mean_obj_pt_dist += std::sqrt(
                std::pow(object_point[j].x - mean_obj_pt.x, 2) +
                std::pow(object_point[j].y - mean_obj_pt.y, 2));
        }

        double sqrt2 = std::sqrt(2);

        Eigen::Matrix3d img_T = Eigen::Matrix3d::Identity();
        img_T(0, 0) = sqrt2 / mean_img_pt_dist;
        img_T(0, 2) = - mean_img_pt.x * sqrt2 / mean_img_pt_dist;
        img_T(1, 1) = sqrt2 / mean_img_pt_dist;
        img_T(1, 2) = - mean_img_pt.y * sqrt2 / mean_img_pt_dist;

        Eigen::Matrix3d obj_T = Eigen::Matrix3d::Identity();
        obj_T(0, 0) = sqrt2 / mean_obj_pt_dist;
        obj_T(0, 2) = - mean_obj_pt.x * sqrt2 / mean_obj_pt_dist;
        obj_T(1, 1) = sqrt2 / mean_obj_pt_dist;
        obj_T(1, 2) = - mean_obj_pt.y * sqrt2 / mean_obj_pt_dist;

        Eigen::MatrixXd L(2 * board_area, 9);
        L.setZero();

        for (int j = 0; j < board_area; j++)
        {
            Eigen::Vector3d img_pt;
            img_pt << image_point[j].x, image_point[j].y, 1.0;
            img_pt = img_T * img_pt;

            Eigen::Vector3d obj_pt;
            obj_pt << object_point[j].x, object_point[j].y, 1.0;
            obj_pt = obj_T * obj_pt;
            obj_pt = obj_pt.transpose();

            L.block<1, 3>(j * 2, 0) = obj_pt;
            L.block<1, 3>(j * 2, 6) = -img_pt(0) * obj_pt;
            L.block<1, 3>(j * 2 + 1, 3) = obj_pt;
            L.block<1, 3>(j * 2 + 1, 6) = -img_pt(1) * obj_pt;
        }

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(L, Eigen::ComputeThinV);
        // std::cout << "Eigen result:" << std::endl
        //           << "W:" << std::endl << svd.singularValues() << std::endl
        //           << "V:" << std::endl << svd.matrixV() << std::endl
        //           << "Last col of V:" << std::endl 
        //           << svd.matrixV().rightCols(1) << std::endl;
        Eigen::VectorXd h = svd.matrixV().rightCols(1);
        Eigen::Matrix3d H;
        H(0, 0) = h(0); H(0, 1) = h(1); H(0, 2) = h(2); 
        H(1, 0) = h(3); H(1, 1) = h(4); H(1, 2) = h(5); 
        H(2, 0) = h(6); H(2, 1) = h(7); H(2, 2) = h(8); 
        // std::cout << "H:" << std::endl << H << std::endl;
        H = img_T.inverse() * H * obj_T;
        Hs[i] = H;
    }

    // Solve matrix B.
    // Extraction of the intrinsic parameters from matrix B.
    Eigen::Matrix3d B;
    Eigen::Matrix3d A;
    {
        Eigen::MatrixXd V(3 * board_count, 6);
        V.setZero();
        auto ConstructVijFromHiHj = [] (Eigen::Vector3d hi, Eigen::Vector3d hj) 
        { 
            Eigen::VectorXd v(6);
            v(0) = hi(0) * hj(0);
            v(1) = hi(0) * hj(1) + hi(1) * hj(0);
            v(2) = hi(1) * hj(1);
            v(3) = hi(2) * hj(0) + hi(0) * hj(2);
            v(4) = hi(2) * hj(1) + hi(1) * hj(2);
            v(5) = hi(2) * hj(2);
            return v;
        };

        for (int i = 0; i < board_count; i++)
        {
            Eigen::Vector3d h0 = Hs[i].col(0);
            Eigen::Vector3d h1 = Hs[i].col(1);
            Eigen::VectorXd v01 = ConstructVijFromHiHj(h0, h1);
            Eigen::VectorXd v00 = ConstructVijFromHiHj(h0, h0);
            Eigen::VectorXd v11 = ConstructVijFromHiHj(h1, h1);
            V.row(i * 3) = v01.transpose();
            V.row(i * 3 + 1) = (v00 - v11).transpose();
        }

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(V, Eigen::ComputeThinV);
        Eigen::VectorXd b = svd.matrixV().rightCols(1);
        B(0, 0) = b(0); B(0, 1) = b(1); B(0, 2) = b(3); 
        B(1, 0) = b(1); B(1, 1) = b(2); B(1, 2) = b(4); 
        B(2, 0) = b(3); B(2, 1) = b(4); B(2, 2) = b(5); 

        double v0 = (b(1) * b(3) - b(0) * b(4)) / (b(0) * b(2) - b(1) * b(1));
        double lambda = b(5) - (b(3) * b(3) + v0 * (b(1) * b(3) - b(0) * b(4))) / b(0);
        double alpha = std::sqrt(lambda / b(0));
        double beta = std::sqrt(lambda * b(0) / (b(0) * b(2) - b(1) * b(1)));
        double gamma = - b(1) * alpha * alpha * beta / lambda;
        double u0 = gamma * v0 / beta - b(3) * alpha * alpha / lambda;
        A.setIdentity();
        A(0, 0) = alpha;
        A(0, 1) = gamma;
        A(0, 2) = u0;
        A(1, 1) = beta;
        A(1, 2) = v0;
        std::cout << "A:" << std::endl << A << std::endl;
    }

    // Solve R, t for each image.
    for (int i = 0; i < board_count; i++)
    {
        Eigen::Vector3d h0 = Hs[i].col(0);
        Eigen::Vector3d h1 = Hs[i].col(1);
        Eigen::Vector3d h2 = Hs[i].col(2);
        Eigen::Matrix3d A_inv = A.inverse();
        double lambda0 = 1.0 / (A_inv * h0).norm();
        double lambda1 = 1.0 / (A_inv * h1).norm();
        // std::cout << "lambda0 = " << lambda0 << std::endl;
        // std::cout << "lambda1 = " << lambda1 << std::endl;
        Eigen::Vector3d r0 = lambda0 * A_inv * h0;
        Eigen::Vector3d r1 = lambda1 * A_inv * h1;
        Eigen::Vector3d r2 = r0.cross(r1);
        Eigen::Vector3d t = lambda0 * A_inv * h2;

        Eigen::Matrix3d R;
        R.col(0) = r0;
        R.col(1) = r1;
        R.col(2) = r2;
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeThinV | Eigen::ComputeThinU);
        R = svd.matrixU() * svd.matrixV().transpose();
        std::cout << "i = " << i << std::endl
                  << "R:" << std::endl << R << std::endl
                  << "t:" << std::endl << t << std::endl;
    }

    return 0;
}
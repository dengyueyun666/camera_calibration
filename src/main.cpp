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
        cout << "Usage :" << endl
             << "./main <config_file>" << endl;
        return -1;
    }

    string config_file = argv[1];
    cout << "config_file : " << config_file << endl;
    cv::FileStorage fs(config_file, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        cout << "Failed to open " << config_file << " file!" << endl;
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
    cout << "Number of images = " << num_image << endl;

    vector<vector<cv::Point2f>> image_points;
    vector<vector<cv::Point3f>> object_points;

    int board_count = 0;
    for (size_t i = 0; i < image_names.size(); i++)
    {
        cv::Mat img = cv::imread(image_path + image_names[i]);
        if (img.empty())
        {
            cerr << "Failed to read " << image_names[i] << " image!" <<endl;
            continue;
        }
        
        vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(img, board_size, corners);

        if (!found)
        {
            cerr << "Failed to find chessboard in " << image_names[i] 
                 << " image!" <<endl;
            continue;
        }
        else
        {
            board_count++;
            image_points.push_back(corners);
            object_points.push_back(vector<cv::Point3f>());
            vector<cv::Point3f> &obj_pts = object_points.back();
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
    cout << "Found " << board_count << " chessboards." << endl;

    for (int i = 0; i < 1; i++)
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
        cout << "Eigen result:" << endl
             << "W:" << endl << svd.singularValues() << endl
             << "V:" << endl << svd.matrixV() << endl
             << "Last col of V:" << endl << svd.matrixV().rightCols(1) << endl;
        Eigen::VectorXd h = svd.matrixV().rightCols(1);
        Eigen::Matrix3d H;
        H(0, 0) = h(0); H(0, 1) = h(1); H(0, 2) = h(2); 
        H(1, 0) = h(3); H(1, 1) = h(4); H(1, 2) = h(5); 
        H(2, 0) = h(6); H(2, 1) = h(7); H(2, 2) = h(8); 
        cout << "H:" << endl << H << endl;
        H = img_T.inverse() * H * obj_T;
        cout << "H:" << endl << H << endl;
    }

    return 0;
}
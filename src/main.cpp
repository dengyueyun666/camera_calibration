#include <iostream>

#include <opencv2/opencv.hpp>

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



    return 0;
}
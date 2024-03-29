#ifndef _CAMERA_CALIBRATION_CALIBRATION_H_
#define _CAMERA_CALIBRATION_CALIBRATION_H_

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

struct Config
{
    cv::Size image_size;
    cv::Size board_size;
    float square_size;
};

// Read images and find chessboards.
int ReadImageAndFindChessboard(
    const std::string& image_path,
    const std::vector<std::string>& image_names,
    const Config& config,
    std::vector<cv::Mat>& images,
    std::vector<std::vector<cv::Point2f>>& pixel_points,
    std::vector<std::vector<cv::Point3f>>& object_points
)
{
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
        
        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(img, config.board_size, corners);

        if (!found)
        {
            std::cerr << "Failed to find chessboard in " << image_names[i] 
                      << " image!" << std::endl;
            continue;
        }
        else
        {
            board_count++;
            images.push_back(img);
            pixel_points.push_back(corners);
            object_points.push_back(std::vector<cv::Point3f>());
            std::vector<cv::Point3f> &obj_pts = object_points.back();
            int board_area = config.board_size.area();
            obj_pts.resize(board_area);
            for (int j = 0; j < board_area; j++)
            {
                obj_pts[j] = cv::Point3f(
                    static_cast<float>(j % config.board_size.width),
                    static_cast<float>(j / config.board_size.width),
                    0.0f) * config.square_size;
            }
        }
    }
    return board_count;
}

// Solve Homography for each image.
void CalculateHomography(
    const std::vector<std::vector<cv::Point2f>>& pixel_points,
    const std::vector<std::vector<cv::Point3f>>& object_points,
    const Config& config,
    std::vector<Eigen::Matrix3d>& Hs
)
{
    int board_count = static_cast<int>(pixel_points.size());
    int board_area = config.board_size.area();
    Hs.resize(board_count);
    for (int i = 0; i < board_count; i++)
    {
        const std::vector<cv::Point2f>& pixel_point = pixel_points[i];
        const std::vector<cv::Point3f>& object_point = object_points[i];

        cv::Point2d mean_pix_pt(0, 0);
        cv::Point2d mean_obj_pt(0, 0);
        for (int j = 0; j < board_area; j++)
        {
            mean_pix_pt.x += pixel_point[j].x;
            mean_pix_pt.y += pixel_point[j].y;

            mean_obj_pt.x += object_point[j].x;
            mean_obj_pt.y += object_point[j].y;
        }
        mean_pix_pt /= board_area;
        mean_obj_pt /= board_area;

        double mean_pix_pt_dist = 0;
        double mean_obj_pt_dist = 0;
        for (int j = 0; j < board_area; j++)
        {
            mean_pix_pt_dist += std::sqrt(
                std::pow(pixel_point[j].x - mean_pix_pt.x, 2) +
                std::pow(pixel_point[j].y - mean_pix_pt.y, 2));

            mean_obj_pt_dist += std::sqrt(
                std::pow(object_point[j].x - mean_obj_pt.x, 2) +
                std::pow(object_point[j].y - mean_obj_pt.y, 2));
        }
        mean_pix_pt_dist /= board_area;
        mean_obj_pt_dist /= board_area;

        double sqrt2 = std::sqrt(2);

        Eigen::Matrix3d pix_T = Eigen::Matrix3d::Identity();
        pix_T(0, 0) = sqrt2 / mean_pix_pt_dist;
        pix_T(0, 2) = - mean_pix_pt.x * sqrt2 / mean_pix_pt_dist;
        pix_T(1, 1) = sqrt2 / mean_pix_pt_dist;
        pix_T(1, 2) = - mean_pix_pt.y * sqrt2 / mean_pix_pt_dist;

        Eigen::Matrix3d obj_T = Eigen::Matrix3d::Identity();
        obj_T(0, 0) = sqrt2 / mean_obj_pt_dist;
        obj_T(0, 2) = - mean_obj_pt.x * sqrt2 / mean_obj_pt_dist;
        obj_T(1, 1) = sqrt2 / mean_obj_pt_dist;
        obj_T(1, 2) = - mean_obj_pt.y * sqrt2 / mean_obj_pt_dist;

        Eigen::MatrixXd L(2 * board_area, 9);
        L.setZero();

        for (int j = 0; j < board_area; j++)
        {
            Eigen::Vector3d pix_pt;
            pix_pt << pixel_point[j].x, pixel_point[j].y, 1.0;
            pix_pt = pix_T * pix_pt;

            Eigen::Vector3d obj_pt;
            obj_pt << object_point[j].x, object_point[j].y, 1.0;
            obj_pt = obj_T * obj_pt;
            obj_pt = obj_pt.transpose();

            L.block<1, 3>(j * 2, 0) = obj_pt;
            L.block<1, 3>(j * 2, 6) = -pix_pt(0) * obj_pt;
            L.block<1, 3>(j * 2 + 1, 3) = obj_pt;
            L.block<1, 3>(j * 2 + 1, 6) = -pix_pt(1) * obj_pt;
        }

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(L, Eigen::ComputeThinV);
        Eigen::VectorXd h = svd.matrixV().rightCols(1);
        Eigen::Matrix3d H;
        H(0, 0) = h(0); H(0, 1) = h(1); H(0, 2) = h(2); 
        H(1, 0) = h(3); H(1, 1) = h(4); H(1, 2) = h(5); 
        H(2, 0) = h(6); H(2, 1) = h(7); H(2, 2) = h(8); 
        H = pix_T.inverse() * H * obj_T;
        H /= H(2, 2);
        Hs[i] = H;
    }
    return;
}

// Calculate intrinsic parameters (alpha, beta, u0, v0)
// Note: u0 and v0 are set to half of image resolution,
//       only alpha and beta are calculated here.
void CalculateIntrinsics(
    const std::vector<Eigen::Matrix3d>& Hs,
    const Config& config,
    Eigen::Matrix3d& camera_matrix
)
{
    double Cx = 0.5 * config.image_size.width;
    double Cy = 0.5 * config.image_size.height;
    int board_count = static_cast<int>(Hs.size());
    Eigen::MatrixXd V(2 * board_count, 2);
    Eigen::VectorXd v(2 * board_count);
    V.setZero();
    v.setZero();

    auto ConstructVijFromHiHjCxCy = [] (Eigen::Vector3d hi, Eigen::Vector3d hj,
                                        double Cx, double Cy) 
    { 
        Eigen::VectorXd vij(2);
        vij(0) = hi(0) * hj(0) - Cx * (hi(0) * hj(2) + hi(2) * hj(0)) +
                 Cx * Cx * hi(2) * hj(2);
        vij(1) = hi(1) * hj(1) - Cy * (hi(1) * hj(2) + hi(2) * hj(1)) +
                 Cy * Cy * hi(2) * hj(2);
        return vij;
    };

    for (int i = 0; i < board_count; i++)
    {
        Eigen::Vector3d h0 = Hs[i].col(0);
        Eigen::Vector3d h1 = Hs[i].col(1);
        Eigen::VectorXd v01 = ConstructVijFromHiHjCxCy(h0, h1, Cx, Cy);
        Eigen::VectorXd v00 = ConstructVijFromHiHjCxCy(h0, h0, Cx, Cy);
        Eigen::VectorXd v11 = ConstructVijFromHiHjCxCy(h1, h1, Cx, Cy);
        V.row(i * 2) = v01.transpose();
        V.row(i * 2 + 1) = (v00 - v11).transpose();
        v(i * 2) = - h0(2) * h1(2);
        v(i * 2 + 1) = h1(2) * h1(2) - h0(2) * h0(2);
    }

    Eigen::VectorXd b = V.colPivHouseholderQr().solve(v);

    double B11 = b(0);
    double B12 = 0;
    double B22 = b(1);
    double B13 = -Cx * B11;
    double B23 = -Cy * B22;
    double B33 = Cx * Cx * B11 + Cy * Cy * B22 + 1;

    double v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12 * B12);
    double lambda = B33 - (B13 * B13 + v0 * (B12 * B13 - B11 * B23)) / B11;
    double alpha = std::sqrt(lambda / B11);
    double beta = std::sqrt(lambda * B11 / (B11 * B22 - B12 * B12));
    double gamma = - B12 * alpha * alpha * beta / lambda;
    double u0 = gamma * v0 / beta - B13 * alpha * alpha / lambda;
    camera_matrix.setIdentity();
    camera_matrix(0, 0) = alpha;
    camera_matrix(0, 1) = gamma; //It will be zero.
    camera_matrix(0, 2) = u0;    //It will be half of image width.
    camera_matrix(1, 1) = beta;
    camera_matrix(1, 2) = v0;    //It will be half of image height;

    return;
}

// Calculate extrinsic parameters (R, t) for each image.
void CalculateExtrinsics(
    const std::vector<Eigen::Matrix3d>& Hs,
    const Eigen::Matrix3d& camera_matrix,
    const Config& config,
    std::vector<Eigen::Matrix3d>& Rs,
    std::vector<Eigen::Vector3d>& ts
)
{
    int board_count = static_cast<int>(Hs.size());
    Rs.resize(board_count);
    ts.resize(board_count);
    Eigen::Matrix3d A_inv = camera_matrix.inverse();
    for (int i = 0; i < board_count; i++)
    {
        Eigen::Vector3d h0 = Hs[i].col(0);
        Eigen::Vector3d h1 = Hs[i].col(1);
        Eigen::Vector3d h2 = Hs[i].col(2);
        double lambda0 = 1.0 / (A_inv * h0).norm();
        double lambda1 = 1.0 / (A_inv * h1).norm();
        Eigen::Vector3d r0 = lambda0 * A_inv * h0;
        Eigen::Vector3d r1 = lambda1 * A_inv * h1;
        Eigen::Vector3d r2 = r0.cross(r1);
        Eigen::Vector3d t = 0.5 * (lambda0 + lambda1) * A_inv * h2;

        // Solve the best rotation matrix to approximate a given 3 x 3 matrix R
        Eigen::Matrix3d R;
        R.col(0) = r0;
        R.col(1) = r1;
        R.col(2) = r2;
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(R, Eigen::ComputeThinV | Eigen::ComputeThinU);
        R = svd.matrixU() * svd.matrixV().transpose();
        Rs[i] = R;
        ts[i] = t;
    }

    return;
}

// Calculate distortion coefficients (k1, k2, p1, p2)
void CalculateDistCoeffs(
    const std::vector<std::vector<cv::Point2f>>& pixel_points,
    const std::vector<std::vector<cv::Point3f>>& object_points,
    const Eigen::Matrix3d& camera_matrix,
    const std::vector<Eigen::Matrix3d>& Rs,
    const std::vector<Eigen::Vector3d>& ts,
    const Config& config,
    Eigen::VectorXd& dist_coeffs
)
{
    int board_count = static_cast<int>(Rs.size());
    int board_area = config.board_size.area();
    Eigen::Matrix3d A_inv = camera_matrix.inverse();
    Eigen::MatrixXd D(2 * board_count * board_area, 4);
    Eigen::VectorXd d(2 * board_count * board_area);
    int row = 0;
    for (int i = 0; i < board_count; i++)
    {
        Eigen::MatrixXd T(3, 4);
        T.leftCols(3) = Rs[i];
        T.rightCols(1) = ts[i];
        const std::vector<cv::Point3f>& object_point = object_points[i];
        const std::vector<cv::Point2f>& pixel_point = pixel_points[i];
        for (int j = 0; j < board_area; j++)
        {
            Eigen::Vector4d obj_pt;
            obj_pt(0) = object_point[j].x;
            obj_pt(1) = object_point[j].y;
            obj_pt(2) = object_point[j].z;
            obj_pt(3) = 1.0;
            Eigen::Vector3d ideal_img_pt = T * obj_pt;
            ideal_img_pt /= ideal_img_pt(2);

            Eigen::Vector3d pix_pt;
            pix_pt(0) = pixel_point[j].x;
            pix_pt(1) = pixel_point[j].y;
            pix_pt(2) = 1.0;
            Eigen::Vector3d real_img_pt = A_inv * pix_pt;
            
            double r2 = std::pow(ideal_img_pt(0), 2) + std::pow(ideal_img_pt(1), 2);
            double r4 = r2 * r2;

            D(row, 0) = ideal_img_pt(0) * r2;
            D(row, 1) = ideal_img_pt(0) * r4;
            D(row, 2) = 2 * ideal_img_pt(0) * ideal_img_pt(1);
            D(row, 3) = r2 + 2 * ideal_img_pt(0) * ideal_img_pt(0);
            d(row) = real_img_pt(0) - ideal_img_pt(0);
            row++;

            D(row, 0) = ideal_img_pt(1) * r2;
            D(row, 1) = ideal_img_pt(1) * r4;
            D(row, 2) = r2 + 2 * ideal_img_pt(1) * ideal_img_pt(1);
            D(row, 3) = 2 * ideal_img_pt(0) * ideal_img_pt(1);
            d(row) = real_img_pt(1) - ideal_img_pt(1);
            row++;
        }
    }
    dist_coeffs = D.colPivHouseholderQr().solve(d);

    return;
}

// Calculate reprojection error.
double CalculateReprojectionError(
    const std::vector<std::vector<cv::Point2f>>& pixel_points,
    const std::vector<std::vector<cv::Point3f>>& object_points,
    const Eigen::Matrix3d& camera_matrix,
    const Eigen::VectorXd& dist_coeffs,
    const std::vector<Eigen::Matrix3d>& Rs,
    const std::vector<Eigen::Vector3d>& ts,
    const Config& config
)
{
    int board_count = static_cast<int>(Rs.size());
    int board_area = config.board_size.area();
    double k1 = dist_coeffs(0);
    double k2 = dist_coeffs(1);
    double p1 = dist_coeffs(2);
    double p2 = dist_coeffs(3);
    double err = 0;
    for (int i = 0; i < board_count; i++)
    {
        const std::vector<cv::Point3f>& object_point = object_points[i];
        const std::vector<cv::Point2f>& pixel_point = pixel_points[i];
        for (int j = 0; j < board_area; j++)
        {
            Eigen::Vector3d pt;
            pt(0) = object_point[j].x;
            pt(1) = object_point[j].y;
            pt(2) = object_point[j].z;
            
            pt = Rs[i] * pt + ts[i];
            pt /= pt(2);

            double x = pt(0);
            double y = pt(1);
            double xy = x * y;
            double r2 = pt(0) * pt(0) + pt(1) * pt(1);
            double r4 = r2 * r2;
            double temp = 1.0 + k1 * r2 + k2 * r4;

            pt(0) = x * temp + 2.0 * p1 * xy + p2 * (r2 + 2.0 * x * x);
            pt(1) = y * temp + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * xy;

            pt = camera_matrix * pt;

            double dx = pixel_point[j].x - pt(0);
            double dy = pixel_point[j].y - pt(1);
            err += dx * dx + dy * dy;
        }
    }
    err /= (board_count * board_area);
    return err;
}

#endif
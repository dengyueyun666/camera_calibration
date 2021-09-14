#ifndef _CAMERA_CALIBRATION_REPROJECTION_FACTOR_H_
#define _CAMERA_CALIBRATION_REPROJECTION_FACTOR_H_

#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

class ReprojectionFactor : public ceres::SizedCostFunction<2, 4, 4, 3, 3>
{
  public:
    ReprojectionFactor(const cv::Point2f& pixel_point, const cv::Point3f& object_point)
        : pixel_point(pixel_point), object_point(object_point) {}
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const
    {
        const double alpha = parameters[0][0];
        const double beta  = parameters[0][1];
        const double u0    = parameters[0][2];
        const double v0    = parameters[0][3];

        const double k1 = parameters[1][0];
        const double k2 = parameters[1][1];
        const double p1 = parameters[1][2];
        const double p2 = parameters[1][3];

        cv::Mat rvec = (cv::Mat_<double>(3, 1) << parameters[2][0],
                                                  parameters[2][1],
                                                  parameters[2][2]);
        cv::Mat R_mat;
        cv::Rodrigues(rvec, R_mat);
        const double R[9] = {
            R_mat.at<double>(0, 0), R_mat.at<double>(0, 1), R_mat.at<double>(0, 2),
            R_mat.at<double>(1, 0), R_mat.at<double>(1, 1), R_mat.at<double>(1, 2),
            R_mat.at<double>(2, 0), R_mat.at<double>(2, 1), R_mat.at<double>(2, 2),
        };

        const double t[3] = {parameters[3][0], parameters[3][1], parameters[3][2]};

        const double op[3] = {object_point.x, object_point.y, object_point.z};
        const double pp[2] = {pixel_point.x, pixel_point.y};

        double xc = R[0] * op[0] + R[1] * op[1] + R[2] * op[2] + t[0];
        double yc = R[3] * op[0] + R[4] * op[1] + R[5] * op[2] + t[1];
        double zc = R[6] * op[0] + R[7] * op[1] + R[8] * op[2] + t[2];

        double x = xc / zc;
        double y = yc / zc;

        double x2 = x  * x; double y2 = y  * y;
        double x3 = x2 * x; double y3 = y2 * y;
        double x4 = x3 * x; double y4 = y3 * y;
        double x5 = x4 * x; double y5 = y4 * y;

        double xi = k2 * x5 + k2 * x * y4 + 2 * k2 * x3 * y2 + k1 * x3
                  + k1 * x * y2 + 3 * p2 * x2 + p2 * y2 + 2 * p1 * x * y + x;
        double yi = k2 * y5 + k2 * x4 * y + 2 * k2 * x2 * y3 + k1 * y3
                  + k1 * x2 * y + 3 * p1 * y2 + p1 * x2 + 2 * p2 * x * y + y;
        
        double u = alpha * xi + u0;
        double v =  beta * yi + v0;

        residuals[0] = u - pp[0];
        residuals[1] = v - pp[1];

        if (jacobians)
        {
            double dxi_dx = 5 * k2 * x4 + k2 * y4 + 6 * k2 * x2 * y2 
                          + 3 * k1 * x2 + k1 * y2 + 6 * p2 * x + 2 * p1 * y + 1;
            double dxi_dy = 4 * k2 * x * y3 + 4 * k2 * x3 * y
                          + 2 * k1 * x * y + 2 * p2 * y + 2 * p1 * x;
            
            double dyi_dx = 4 * k2 * x3 * y + 4 * k2 * x * y3
                          + 2 * k1 * x * y + 2 * p1 * x + 2 * p2 * y;
            double dyi_dy = 5 * k2 * y4 + k2 * x4 + 6 * k2 * x2 * y2
                          + 3 * k1 * y2 + k1 * x2 + 6 * p1 * y + 2 * p2 * x + 1;

            double dxi_dk1 = x3 + x * y2;
            double dxi_dk2 = x5 + x * y4 + 2 * x3 * y2;
            double dxi_dp1 = 2 * x * y;
            double dxi_dp2 = y2 + 3 * x2;

            double dyi_dk1 = y3 + x2 * y;
            double dyi_dk2 = y5 + x4 * y + 2 * x2 * y3;
            double dyi_dp1 = x2 + 3 * y2;
            double dyi_dp2 = 2 * x * y;

            double dx_dxc = 1 / zc, dx_dyc =      0, dx_dzc = - xc / (zc * zc);
            double dy_dxc =      0, dy_dyc = 1 / zc, dy_dzc = - yc / (zc * zc);

            double rx = xc - t[0];
            double ry = yc - t[1];
            double rz = zc - t[2];

            double dxc_dw0 =   0, dxc_dw1 =  rz, dxc_dw2 = -ry;
            double dyc_dw0 = -rz, dyc_dw1 =   0, dyc_dw2 =  rx;
            double dzc_dw0 =  ry, dzc_dw1 = -rx, dzc_dw2 =   0;

            double dxc_dt0 = 1, dyc_dt1 = 1, dzc_dt2 = 1;

            double dx_dw0 = dx_dxc * dxc_dw0 + dx_dyc * dyc_dw0 + dx_dzc * dzc_dw0;
            double dx_dw1 = dx_dxc * dxc_dw1 + dx_dyc * dyc_dw1 + dx_dzc * dzc_dw1;
            double dx_dw2 = dx_dxc * dxc_dw2 + dx_dyc * dyc_dw2 + dx_dzc * dzc_dw2;
            
            double dy_dw0 = dy_dxc * dxc_dw0 + dy_dyc * dyc_dw0 + dy_dzc * dzc_dw0;
            double dy_dw1 = dy_dxc * dxc_dw1 + dy_dyc * dyc_dw1 + dy_dzc * dzc_dw1;
            double dy_dw2 = dy_dxc * dxc_dw2 + dy_dyc * dyc_dw2 + dy_dzc * dzc_dw2;

            double dxi_dw0 = dxi_dx * dx_dw0 + dxi_dy * dy_dw0;
            double dxi_dw1 = dxi_dx * dx_dw1 + dxi_dy * dy_dw1;
            double dxi_dw2 = dxi_dx * dx_dw2 + dxi_dy * dy_dw2;

            double dyi_dw0 = dyi_dx * dx_dw0 + dyi_dy * dy_dw0;
            double dyi_dw1 = dyi_dx * dx_dw1 + dyi_dy * dy_dw1;
            double dyi_dw2 = dyi_dx * dx_dw2 + dyi_dy * dy_dw2;

            double dx_dt0 = dx_dxc * dxc_dt0;
            double dx_dt1 = dx_dyc * dyc_dt1;
            double dx_dt2 = dx_dzc * dzc_dt2;

            double dy_dt0 = dy_dxc * dxc_dt0;
            double dy_dt1 = dy_dyc * dyc_dt1;
            double dy_dt2 = dy_dzc * dzc_dt2;

            double dxi_dt0 = dxi_dx * dx_dt0 + dxi_dy * dy_dt0;
            double dxi_dt1 = dxi_dx * dx_dt1 + dxi_dy * dy_dt1;
            double dxi_dt2 = dxi_dx * dx_dt2 + dxi_dy * dy_dt2;

            double dyi_dt0 = dyi_dx * dx_dt0 + dyi_dy * dy_dt0;
            double dyi_dt1 = dyi_dx * dx_dt1 + dyi_dy * dy_dt1;
            double dyi_dt2 = dyi_dx * dx_dt2 + dyi_dy * dy_dt2;

            if (jacobians[0])
            {
                // derivative of u with respect of
                // alpha, beta, u0, v0
                jacobians[0][0] = xi;
                jacobians[0][1] =  0;
                jacobians[0][2] =  1;
                jacobians[0][3] =  0;

                // derivative of v with respect of
                // alpha, beta, u0, v0
                jacobians[0][4] =  0;
                jacobians[0][5] = yi;
                jacobians[0][6] =  0;
                jacobians[0][7] =  1;
            }
            if (jacobians[1])
            {
                // derivative of u with respect of
                // k1, k2, p1, p2
                jacobians[1][0] = alpha * dxi_dk1;
                jacobians[1][1] = alpha * dxi_dk2;
                jacobians[1][2] = alpha * dxi_dp1;
                jacobians[1][3] = alpha * dxi_dp2;

                // derivative of v with respect of
                // k1, k2, p1, p2
                jacobians[1][4] = beta * dyi_dk1;
                jacobians[1][5] = beta * dyi_dk2;
                jacobians[1][6] = beta * dyi_dp1;
                jacobians[1][7] = beta * dyi_dp2;
            }
            if (jacobians[2])
            {
                // derivative of u with respect of
                // w0, w1, w2
                jacobians[2][0] = alpha * dxi_dw0;
                jacobians[2][1] = alpha * dxi_dw1;
                jacobians[2][2] = alpha * dxi_dw2;

                // derivative of v with respect of
                // w0, w1, w2
                jacobians[2][3] = beta * dyi_dw0;
                jacobians[2][4] = beta * dyi_dw1;
                jacobians[2][5] = beta * dyi_dw2;
            }
            if (jacobians[3])
            {
                // derivative of u with respect of
                // t0, t1, t2
                jacobians[3][0] = alpha * dxi_dt0;
                jacobians[3][1] = alpha * dxi_dt1;
                jacobians[3][2] = alpha * dxi_dt2;

                // derivative of v with respect of
                // t0, t1, t2
                jacobians[3][3] = beta * dyi_dt0;
                jacobians[3][4] = beta * dyi_dt1;
                jacobians[3][5] = beta * dyi_dt2;
            }
        }

        return true;
    }

    cv::Point2f pixel_point;
    cv::Point3f object_point;
};

#endif

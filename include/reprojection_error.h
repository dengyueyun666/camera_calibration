#ifndef _CAMERA_CALIBRATION_REPROJECTION_ERROR_H_
#define _CAMERA_CALIBRATION_REPROJECTION_ERROR_H_

#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

struct ReprojectionError
{
    ReprojectionError(cv::Point2f pixel_point_, cv::Point3f object_point_)
        : pixel_point(pixel_point_), object_point(object_point_) {}
    
    template <typename T>
    bool operator()(const T* const rvec,
                    const T* const tvec,
                    const T* const cam_mat,
                    const T* const dist_coes,
                    // const T* const obj_pt,
                    T* residuals) const
    {
        T obj_pt[3];
        obj_pt[0] = T(object_point.x);
        obj_pt[1] = T(object_point.y);
        obj_pt[2] = T(object_point.z);

        // World coordinates -> Camera coordinates
        T p[3];
        ceres::AngleAxisRotatePoint(rvec, obj_pt, p);
        p[0] += tvec[0];
        p[1] += tvec[1];
        p[2] += tvec[2];

        // Camera coordinates -> Undistorted normalized image coordinates
        T un_img_pt_x = p[0] / p[2];
        T un_img_pt_y = p[1] / p[2];

        // Undistorted normalized image coordinates -> 
        //   Distorted normalized image coordinates
        T r2 = un_img_pt_x * un_img_pt_x + un_img_pt_y * un_img_pt_y;
        T r4 = r2 * r2;
        const T& k1 = dist_coes[0];
        const T& k2 = dist_coes[1];
        const T& p1 = dist_coes[2];
        const T& p2 = dist_coes[3];
        T img_pt_x = un_img_pt_x * (1.0 + k1 * r2 + k2 * r4) +
                     2.0 * p1 * un_img_pt_x * un_img_pt_y +
                     p2 * (r2 + 2.0 * un_img_pt_x * un_img_pt_x);
        T img_pt_y = un_img_pt_y * (1.0 + k1 * r2 + k2 * r4) +
                     2.0 * p2 * un_img_pt_x * un_img_pt_y +
                     p1 * (r2 + 2.0 * un_img_pt_y * un_img_pt_y);
        
        // Distorted normalized image coordinates -> Pixel coordinates
        // cam_mat[0,1,2,3,4] are alpha, beta, gamma, u0 and v0.
        T pix_pt_x = cam_mat[0] * img_pt_x + 
                     cam_mat[2] * img_pt_y +
                     cam_mat[3];
        T pix_pt_y = cam_mat[1] * img_pt_y + cam_mat[4];

        // Error
        residuals[0] = pix_pt_x - T(pixel_point.x);
        residuals[1] = pix_pt_y - T(pixel_point.y);

        return true;
    }

    static ceres::CostFunction* Create(const cv::Point2f pixel_point,
                                       const cv::Point3f object_point)
    {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 3, 5, 4>(
            new ReprojectionError(pixel_point, object_point)));
    }

    cv::Point2f pixel_point;
    cv::Point3f object_point;
};

#endif
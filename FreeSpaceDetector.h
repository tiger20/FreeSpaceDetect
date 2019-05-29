#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <eigen3/Eigen/Dense>

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "tic_toc.h"
using namespace std;
using namespace cv;
using namespace Eigen;

//const double PI = 3.14159265;
extern bool Debug;
struct LidarParam
{
    int N_SCAN;
    int Horizon_SCAN;
    float ang_res_x;
    float ang_res_y;
    float ang_bottom;
    float sensorMountAngle;
    int groundScanInd;
    LidarParam(){
        N_SCAN = 64;
        Horizon_SCAN = 4500;
        ang_res_x = 0.08;
        ang_res_y = 26.9/63;
        ang_bottom = -24.9f;
        groundScanInd = 60;
        sensorMountAngle = 0.0;
    }
    LidarParam(int ns, int hs, float rx, float ry, float aglb, int grs, float mountAngle){
        N_SCAN = ns;
        Horizon_SCAN = hs;
        ang_res_x = rx;
        ang_res_y = ry;
        ang_bottom = aglb;
        groundScanInd = grs;
        sensorMountAngle = mountAngle;
    }
};
struct Edge{
    Vector3d coeffs;
    Vector2d startpt;
    Vector2d endpt;
};
class FreeSpaceDetector{
public:
    LidarParam ldr_param;
    pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudRaw;
    pcl::PointCloud<pcl::PointXYZI>::Ptr sparseCloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr segResult;
    Vector3d plane;
    Mat frame;
    Mat bev;
    Mat labels;
    int sprsWidth;
    int sprsHeight;
    double fov;
    FreeSpaceDetector();
    void setData(Mat& img, pcl::PointCloud<pcl::PointXYZI>::Ptr cloud);
    void setParam(LidarParam param);
    bool getFreeSpace();
    void projectCurveOnImage(Mat& img, const Matrix3d& H, Vector3d coeffs0, Point2f startpt0, Point2f endpt0,
                                            Vector3d coeffs1, Point2f startpt1, Point2f endpt1);
    Point2f projectPoint(const Matrix3d &H, const Vector3d &point);
    TicToc clock;
    bool inBorder(const Mat &image, const Point2f &pt);
    double fcurve(const Vector3d X, double x) { return X(0) * x * x + X(1) * x + X(2); }

private:
    Edge leftEdge, rightEdge;
    void downSample(int gap);
    Matrix3d computeH();//plane modle: z = a_1x + a_2y + a_3
    void segmentRoad();
    void rangeConv(const Mat& input, Mat& output, int flag);
    void genFreeArea(Mat& input, Mat& output, bool use_pad_up);
    void threshold_range(Mat &range, int ths);
    void drawCurve(Mat& img, Vector3d coeffs0, int starty0, int endy0, Vector3d coeffs1, int starty1, int endy1);
    int fitRoadEdges(const Mat &area, Mat &output, Vector3d &coeffs0, Point2f& startpt0, Point2f& endpt0, Vector3d& coeffs1, Point2f& startpt1, Point2f& endpt1);
    float scale;
};
#include "FreeSpaceDetector.h"

bool Debug = false;
const int PAD_TOP = 150;
const int PAD_DOWN = 250;
const int CENT_Y = 395;
const int CENT_X = 256;
const int MAP_WIDTH = 512;
const int LINE_TOP = 150;
const int LINE_DOWN = 350;

FreeSpaceDetector::FreeSpaceDetector()
{
    sparseCloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
    fov = 126.0;
    bev = Mat::zeros(512, 512, CV_8UC1);
}
void FreeSpaceDetector::setData(Mat& img, pcl::PointCloud<pcl::PointXYZI>::Ptr cloud){
    frame = img;
    laserCloudRaw = cloud;
}
void FreeSpaceDetector::setParam(LidarParam param){
    ldr_param = param;
}
void FreeSpaceDetector::downSample(int gap){
    int sz = laserCloudRaw->points.size();
    sprsHeight = 64;
    sprsWidth = static_cast<int>(fov / (0.08 * gap) + 0.5) + 1;

    labels = Mat::zeros(sprsHeight, sprsWidth, CV_8UC1);
    sparseCloud->points.resize(sprsHeight * sprsWidth);
    pcl::PointXYZI nanPoint;
    nanPoint.x = -1;
    nanPoint.y = -1;
    nanPoint.z = -1;
    nanPoint.intensity = -1;
    std::fill(sparseCloud->points.begin(), sparseCloud->points.end(), nanPoint);

    pcl::PointXYZI pt;
    for(int i = 0; i < sz; ++i){
        pt = laserCloudRaw->points[i];
        if(pt.x >= 5){
            double angle = atan2(pt.y, pt.x) * 180.0 / M_PI;
            if (angle < fov/2 && angle > -fov/2)
            {
                int nth = static_cast<int>((fov / 2 - angle) / (0.08 * gap) + 0.5);
                double verAngle = atan2(pt.z, sqrt(pt.x * pt.x + pt.y * pt.y)) * 180 / M_PI;
                int rowid = static_cast<int>((verAngle - ldr_param.ang_bottom) / ldr_param.ang_res_y + 0.5);
                int idx = rowid * sprsWidth + nth;
                sparseCloud->points[idx] = pt;
                labels.at<uchar>(rowid, nth) = 1;
            }
        }
    }
    // imshow("img", labels * 255);
    // waitKey(0);
}
void FreeSpaceDetector::segmentRoad(){
    size_t lowerInd, upperInd;
    float diffX, diffY, diffZ, angle;
    vector<Vector3d> samples;
    int sampleGap = 10;
    for (int col = 0; col < sprsWidth; ++col)
    {
        for(int row = 0; row < ldr_param.groundScanInd; ++row){
            lowerInd = col + row*sprsWidth;
            upperInd = col + (row+1)*sprsWidth;
            if(labels.at<uchar>(row,col) == 0 || labels.at<uchar>(row+1,col) == 0)
                continue;
            diffX = sparseCloud->points[upperInd].x - sparseCloud->points[lowerInd].x;
            diffY = sparseCloud->points[upperInd].y - sparseCloud->points[lowerInd].y;
            diffZ = sparseCloud->points[upperInd].z - sparseCloud->points[lowerInd].z;
            angle = atan2(diffZ, sqrt(diffX*diffX + diffY*diffY) ) * 180 / M_PI;
            if (abs(angle - ldr_param.sensorMountAngle) <= 10){
                labels.at<uchar>(row,col) = 2;
                labels.at<uchar>(row+1,col) = 2;
                if(lowerInd % sampleGap == 0)
                    samples.emplace_back(Vector3d(sparseCloud->points[lowerInd].x,
                                                  sparseCloud->points[lowerInd].y,
                                                  sparseCloud->points[lowerInd].z));
            }
        }
    }
    //cout << samples.size() << endl;
    const int l = samples.size();
    assert(l > 80);
    sort(samples.begin(), samples.end(), [&](Vector3d &a, Vector3d &b) { return a.z() < b.z(); });
    const int N = 50;
    Eigen::MatrixXd P1(N, 3), P2(N, 1), A, b;
    int start = l / 2 - N / 2;
    for (int i = start; i < start + N; ++i){
        P1.block<1, 3>(i-start, 0) = Vector3d(samples[i].x(), samples[i].y(), 1);
        P2(i-start,0) = samples[i].z();
    }
    A = P1.transpose() * P1;
    b = P1.transpose() * P2;
    Vector3d X = A.ldlt().solve(b);
    plane = X;
    double ths = 0.12;
    if (Debug)
        segResult.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointXYZRGB point;
    scale = 7.0;
    for (int i = 0; i < sprsHeight; i++)
    {
        for (int j = 0; j < sprsWidth; j++){
            if (labels.at<uchar>(i, j) != 0){
                int idx = j + i * sprsWidth;
                point.x = sparseCloud->points[idx].x;
                point.y = sparseCloud->points[idx].y;
                point.z = sparseCloud->points[idx].z;
                double error = point.z - (X(0) * point.x + X(1) * point.y + X(2));
                if(error > ths){
                    float xp, yp, xx, yy;
                    xp = point.x;
                    yp = point.y;
                    xx = CENT_Y - xp * scale;
                    yy = CENT_X - yp * scale;
                    if (xx >= 0 && xx < 512 && yy >= 0 && yy < 512)
                        bev.at<uchar>(static_cast<int>(xx + 0.5), static_cast<int>(yy + 0.5)) = 255;
                }
                if(Debug){
                    uint32_t rgb;
                    if (error > ths)
                        rgb = (static_cast<uint32_t>(255) << 16 | static_cast<uint32_t>(0) << 8 | static_cast<uint32_t>(0));
                    else
                        rgb = (static_cast<uint32_t>(0) << 16 | static_cast<uint32_t>(255) << 8 | static_cast<uint32_t>(0));
                    point.rgb = *reinterpret_cast<float *>(&rgb);
                    segResult->points.push_back(point);
                }
            }
        }
    }
    //imwrite("bev.jpg", bev);
    // imshow("bev", bev);
    // waitKey(0);
}

void FreeSpaceDetector::rangeConv(const Mat& input, Mat& output, int flag){
    output = input.clone();
    int m1 = 3;
	int m3 = m1;
	int m2 = 1;
	int m4 = 3;
    uchar *pd = output.data;
    for (int i = 0; i < output.rows; ++i)
        for (int j = 0; j < output.cols; ++j)
            pd[j + i * output.cols] = 255 - pd[j + i * output.cols];

    uchar *p1 = output.data;
    for (int i = 1; i < 511; i++){
        for (int j = 1; j < 511; j++){
            if (p1[512 * i + j] < 254){
                if (p1[512 * i + j] + m1 < p1[512 * (i + 1) + (j - 1)])
                    p1[512 * (i + 1) + (j - 1)] = p1[512 * i + j] + m1;
                if (p1[512 * i + j] + m2 < p1[512 * (i + 1) + j])
                    p1[512 * (i + 1) + j] = p1[512 * i + j] + m2;
                if (p1[512 * i + j] + m3 < p1[512 * (i + 1) + (j + 1)])
                    p1[512 * (i + 1) + (j + 1)] = p1[512 * i + j] + m3;
                if (p1[512 * i + j] + m4 < p1[512 * i + (j + 1)])
                    p1[512 * i + (j + 1)] = p1[512 * i + j] + m4;
            }
        }
	}
	for (int i = 510; i > 0; i--) {
		for (int j = 510; j > 0; j--) {
			if (p1[512 * i + j] < 254) {
				if (p1[512 * i + j] + m1 < p1[512 * (i - 1) + (j - 1)])
					p1[512 * (i - 1) + (j - 1)] = p1[512 * i + j] + m1;
				if (p1[512 * i + j] + m2 < p1[512 * (i - 1) + j])
					p1[512 * (i - 1) + j] = p1[512 * i + j] + m2;
				if (p1[512 * i + j] + m3 < p1[512 * (i - 1) + (j + 1)])
					p1[512 * (i - 1) + (j + 1)] = p1[512 * i + j] + m3;
				if (p1[512 * i + j] + m4 < p1[512 * i + (j - 1)])
					p1[512 * i + (j - 1)] = p1[512 * i + j] + m4;
			}
		}
	}
}
void FreeSpaceDetector::threshold_range(Mat& range, int ths){
    uchar *pd = range.data;
    for(int i = 0; i < range.rows; ++i)
        for(int j = 0; j < range.cols; ++j)
            if (pd[j + i * range.cols] <= ths)
                pd[j + i * range.cols] = 255;
            else
                pd[j + i * range.cols] = 0;
}
void FreeSpaceDetector::genFreeArea(Mat& input, Mat& output, bool use_pad_up){
    int left_start = 0;
	int right_start = 0;
	int up_start = 0;
    output = Mat::zeros(input.rows, input.cols, CV_8UC1);
    uchar *p0, *p1;
    p0 = input.data;
    p1 = output.data;
    for (int ii = CENT_Y; ii >= 0; ii--){
        int det = CENT_Y - ii;
		float k = det / (float)(CENT_X - 0);
		for (int tmp_j = CENT_X; tmp_j >= 0; tmp_j--){
			int tmp_i = CENT_Y - (CENT_X - tmp_j) * k;
			if (p0[MAP_WIDTH * tmp_i + tmp_j] == 0){
				if (left_start == 1){
					p1[MAP_WIDTH * tmp_i + tmp_j] = 255;
				}
			} else {
				left_start = 1;
				break;
			}
		}
		k = det / (float)(MAP_WIDTH - CENT_X);
		for (int tmp_j = CENT_X; tmp_j < MAP_WIDTH; tmp_j++) {
			int tmp_i = CENT_Y - (tmp_j - CENT_X) * k;
			if (p0[MAP_WIDTH * tmp_i + tmp_j] == 0) {
				if (right_start == 1) {
					p1[MAP_WIDTH * tmp_i + tmp_j] = 255;
				}
			} 
			else {
				right_start = 1;
				break;
			}
		}
	}
	for (int jj = 0; jj < MAP_WIDTH; jj++) {
		int det = CENT_X - jj;
		float k = det / float(CENT_Y - 0);
		for (int tmp_i = CENT_Y; tmp_i >=0; tmp_i--) {
			int tmp_j = CENT_X - (CENT_Y - tmp_i) * k;
			if (p0[MAP_WIDTH * tmp_i + tmp_j] == 0){
				if (up_start == 1) {
					p1[MAP_WIDTH * tmp_i + tmp_j] = 255;
				}
			}else{
				up_start = 1;
				break;
			}
		}
	}
    
    if (use_pad_up){
        cout << "use pad up" << endl;
        for (int jj = 0; jj < MAP_WIDTH; jj++) {
			int y_min = -1;
			for (int ii = PAD_DOWN; ii >= PAD_TOP; ii--) {
				if (p1[MAP_WIDTH * ii + jj] == 255) {
					y_min = ii;
				}
			}
			if (y_min > PAD_TOP) {
				for (int ii = y_min - 1; ii >= 0; ii--) {
					if (p0[MAP_WIDTH * ii + jj] == 0) {
						p1[MAP_WIDTH * ii + jj] = 200;
					} else {
						break;
					}
				}
			}
		}
	}
}
int FreeSpaceDetector::fitRoadEdges(const Mat &area, Mat &output, Vector3d &coeffs0, Point2f& startpt0, Point2f& endpt0, Vector3d &coeffs1, Point2f& startpt1, Point2f& endpt1){
    uchar *p0 = area.data;
    int left_x = -1;
	int right_x = -1;
    vector<Point2f> leftline, rightline;
	for (int jj = 0; jj < MAP_WIDTH; jj++) {
		if (p0[MAP_WIDTH * LINE_TOP + jj] > 0) {
			if (left_x == -1){
				left_x = jj;
			}
			right_x = jj;
		}
	}
	if (left_x== -1 || right_x == -1) {
		return 0;
	}
    leftline.emplace_back(Point2f(LINE_TOP, left_x));
    rightline.emplace_back(Point2f(LINE_TOP, right_x));
    for (int ii = LINE_TOP + 1; ii <= LINE_DOWN; ii++) {
		int left_find = 0;
		int right_find = 0;
		for (int jj = left_x - 1; jj >= left_x - 10 && jj >= 0; jj--) {
			if (p0[MAP_WIDTH * ii + jj] > 0){
				left_find = 1;
				left_x = left_x - 1;
				break;
			}
		}
		if (left_find == 0) {
			for (int jj = left_x; jj < left_x + 10 && jj < MAP_WIDTH; jj++) {
				if (p0[MAP_WIDTH * ii + jj] > 0){
					left_find = 1;
					if (left_x != jj) {
						left_x = left_x + 1;
					}
					break;
				}
			}
		}
		for (int jj = right_x + 1; jj < right_x + 10 && jj < MAP_WIDTH; jj++) {
			if (p0[MAP_WIDTH * ii + jj] > 0) {
				right_find = 1;
				right_x = right_x + 1;
				break;
			}
		}
		if (right_find == 0) {
			for (int jj = right_x; jj > right_x - 10 && jj >= 0; jj--){
				if (p0[MAP_WIDTH * ii + jj] > 0){
					right_find = 1;
					if (right_x != jj) {
						right_x = right_x -1;
					}
					break;
				}
			}
		}
		if (left_find == 0 || right_find == 0 || left_x >= right_x) {
			return 0;
		}
		if ((ii & 3) == 0) {
			leftline.emplace_back(Point2f(ii, left_x));
			rightline.emplace_back(Point2f(ii, right_x));
		}
	}
    // int starty0 = leftline[0].x;
    // int endy0 = leftline.back().x;
    // int starty1 = rightline[0].x;
    // int endy1 = rightline.back().x;

    MatrixXd P_l0(leftline.size(), 3), P_l1(leftline.size(), 1);
    MatrixXd P_r0(rightline.size(), 3), P_r1(rightline.size(), 1);
    double x,y;
    for (int i = 0; i < leftline.size(); ++i){
        x = leftline[i].x;
        y = leftline[i].y;
        P_l0.block<1,3>(i,0) = Vector3d(x*x, x, 1);
        P_l1(i,0) = y;

        x = rightline[i].x;
        y = rightline[i].y;
        P_r0.block<1,3>(i,0) = Vector3d(x*x, x, 1);
        P_r1(i,0) = y;
    }

    double offset = 3.0;
    MatrixXd A, b;
    Vector3d X;
    A = P_l0.transpose()*P_l0;
    b = P_l0.transpose()*P_l1;
    X = A.ldlt().solve(b);
    X(2) -= offset;
    coeffs0 = X;

    x = leftline[0].x;
    startpt0.x = x;
    startpt0.y = X(0) * x * x + X(1) * x + X(2);
    x = leftline.back().x;
    endpt0.x = x;
    endpt0.y = X(0) * x * x + X(1) * x + X(2);

    A = P_r0.transpose() * P_r0;
    b = P_r0.transpose()*P_r1;
    X = A.ldlt().solve(b);
    X(2) += offset;
    coeffs1 = X;

    x = rightline[0].x;
    startpt1.x = x;
    startpt1.y = X(0) * x * x + X(1) * x + X(2);
    x = rightline.back().x;
    endpt1.x = x;
    endpt1.y = X(0) * x * x + X(1) * x + X(2);
}
void FreeSpaceDetector::drawCurve(Mat& img, Vector3d coeffs0, int starty0, int endy0, Vector3d coeffs1, int starty1, int endy1){
    vector<Point2f> pts0, pts1;
    Point2f lastpt, curpt;
    for(int i = starty0; i <= endy0; ++i){
        float x = coeffs0[0]*i*i + coeffs0[1]*i+coeffs0[2];
        curpt = Point2f(x, static_cast<float>(i));
        if(i > starty0)
            line(img, lastpt, curpt, Scalar(0, 255, 0), 1.5);
        lastpt = curpt;
    }
    for(int i = starty1; i <= endy1; ++i){
        float x = coeffs1[0]*i*i + coeffs1[1]*i+coeffs1[2];
        curpt = Point2f(x, static_cast<float>(i));
        if(i > starty1)
            line(img, lastpt, curpt, Scalar(0, 255, 0), 1.5);
        lastpt = curpt;
    }
}
Matrix3d FreeSpaceDetector::computeH(){
    Matrix<double, 4,3> plane_to_lidar;
    plane_to_lidar << -1/scale, 0, CENT_Y/scale,
                       0, -1/scale, CENT_X/scale,
                       -plane[0]/scale, -plane[1]/scale, (plane[0]*CENT_Y+plane[1]*CENT_X)/scale+plane[2],
                       0, 0, 1;

    Matrix<double, 3,4> P_lidar_to_img;
    P_lidar_to_img << 609.6954, -721.4216, -1.2513, -123.0418,
                      180.3842, 7.6448, -719.6515, -101.0167,
                      0.9999,   0.0001, 0.0105,   -0.2694;
    return (P_lidar_to_img*plane_to_lidar);
}
Point2f FreeSpaceDetector::projectPoint(const Matrix3d &H, const Vector3d &point){
    Vector3d res = H * point;
    res /= res.z();
    return Point2f(res.x(), res.y());
}
bool FreeSpaceDetector::inBorder(const Mat& image, const Point2f& pt){
    if(pt.x >=0 && pt.x < image.cols && pt.y >= 0 && pt.y < image.rows)
        return true;
    else
        return false;
}
void FreeSpaceDetector::projectCurveOnImage(Mat& img, const Matrix3d& H, 
                                            Vector3d coeffs0, Point2f startpt0, Point2f endpt0,
                                            Vector3d coeffs1, Point2f startpt1, Point2f endpt1){
    vector<Point> ptsl, ptsr;
    Point2f p;
    for (int i = static_cast<int>(startpt0.x + 0.5); i <= static_cast<int>(endpt0.x + 0.5); ++i){
        double y = fcurve(coeffs0, i);
        p = projectPoint(H, Vector3d(i, y, 1));
        ptsl.emplace_back(Point(int(p.x+0.5), int(p.y + 0.5)));

        y = fcurve(coeffs1, i);
        p = projectPoint(H, Vector3d(i, y, 1));
        ptsr.emplace_back(Point(int(p.x+0.5), int(p.y + 0.5)));
    }
    int len = ptsl.size();
    Point *ptsall = new Point[2*len];
    for (int i = 0; i < len; ++i)
        ptsall[i] = ptsl[i];
    for (int i = 0; i < len; ++i)
        ptsall[i + len] = ptsr[len - i - 1];
    const Point *pts[] = {ptsall, ptsall + len};
    int npts[2];
    npts[0] = npts[1] = len;
    Mat mask = Mat::zeros(img.size(), CV_8UC1);
    int npts2[1];
    npts2[0] = 2 * len;
    fillPoly(mask, pts, npts2, 1, Scalar(255), 8, 0, Point());

    uchar *data0 = frame.data;
    uchar *data1 = mask.data;
    for(int i = 0; i < frame.rows; ++i)
        for(int j = 0; j < frame.cols; ++j){
            int idx0 = 3 * j + i * 3 * frame.cols;
            int idx1 = j + i * frame.cols;
            if(data1[idx1] > 50){
                float s = 0.5;
                data0[idx0] *= s;
                data0[idx0 + 1] = data0[idx0 + 1] * s + 255 * s;
                data0[idx0 + 2] *= s;
            }
        }
    // imshow("mask", mask);
    // waitKey();
    polylines(img, pts, npts, 2, false, Scalar(0, 255, 255), 3, 8, 0);
    imwrite("freespace.jpg", img);
}
bool FreeSpaceDetector::getFreeSpace(){
    //clock.tic();
    downSample(5);
    // double dura1 = clock.toc();
    // cout<<"Downsample : "<<dura1<<"ms"<<"    ";

    //clock.tic();
    segmentRoad();
    //double dura2 = clock.toc();
    //cout<<"Segment Road : "<<dura2<<"ms"<<"    ";
    Mat range;
    rangeConv(bev, range, 1);
    //imshow("rg1", range);
    threshold_range(range, 20);
    Mat freearea;
    genFreeArea(range, freearea, false);
    Vector3d coeffs0, coeffs1;
    Point2f s0,e0,s1,e1;
    fitRoadEdges(freearea, bev, coeffs0, s0, e0, coeffs1, s1, e1);
    Matrix3d H = computeH();
    projectCurveOnImage(frame, H, coeffs0, s0, e0, coeffs1, s1, e1);

    //cout<<s0<<" "<<e0<<endl;
    Mat show = bev.clone();
    cvtColor(show, show, CV_GRAY2BGR);
    drawCurve(show, coeffs0, s0.x, e0.x, coeffs1, s1.x, e1.x);
    imwrite("bev.jpg", show);
    imshow("frme", frame);
    waitKey(0);
    return true;
}
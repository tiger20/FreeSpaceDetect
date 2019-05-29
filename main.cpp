#include "FreeSpaceDetector.h"
#include "tic_toc.h"
#include <fstream>
#include <iostream>

using namespace std;
pcl::PointCloud<pcl::PointXYZI>::Ptr readfile(string filename){
    std::fstream input(filename, std::ios::in | std::ios::binary);
    if(!input.good()){
        std::cerr << "Could not read file: " << std::endl;
        exit(EXIT_FAILURE);
    }
    input.seekg(0, std::ios::beg);
    pcl::PointCloud<pcl::PointXYZI>::Ptr points (new pcl::PointCloud<pcl::PointXYZI>);
    int i;
    for (i = 0; input.good() && !input.eof(); i++)
    {
        pcl::PointXYZI point;
        input.read((char *) &point.x, 3*sizeof(float));
        input.read((char *) &point.intensity, sizeof(float));
        points->push_back(point);
    }
    input.close();
    return points;
}
boost::shared_ptr<pcl::visualization::PCLVisualizer> simpleVis (pcl::PointCloud<pcl::PointXYZI>::ConstPtr cloud)
{
    // --------------------------------------------
    // -----Open 3D viewer and add point cloud-----
    // --------------------------------------------
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    viewer->addPointCloud<pcl::PointXYZI> (cloud, "sample cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
    return (viewer);
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> rgbVis (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}

int main(){
    //TicToc clock;

    FreeSpaceDetector det;
    LidarParam param;
    det.setParam(param);
    string datapath = "/Volumes/Orange/DataSet/paper_data/2011_09_26_93/2011_09_26_drive_0093_sync";
    Mat img = imread(datapath + "/image_02/data/0000000302.png");
    pcl::PointCloud<pcl::PointXYZI>::Ptr data = readfile(datapath + "/velodyne_points/data/0000000302.bin");

    //clock.tic();
    det.setData(img, data);
    det.getFreeSpace();
    //cout<<clock.toc()<<" ms"<<endl;
    if(Debug){
        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
        viewer = rgbVis(det.segResult);
        while (!viewer->wasStopped ())
        {
            viewer->spinOnce (100);
            boost::this_thread::sleep (boost::posix_time::microseconds (100000));
        }
    }
    return 0;
}
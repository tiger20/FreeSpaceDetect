# FreeSpaceDetect
Free space detection for automated vehicles

Prerequisites:
+ OpenCV
+ PCL
+ Eigen

It's an implementation of a simple idea. I want to efficiently plan a region where the vehicle is free to go. The algorithm only uses a small subset of the Kitti lidar data per frame since most of the point cloud is redundant.

The idea is quite simple, first we do the segmentation to locate the road plane.

<center>
    <img src="https://github.com/tiger20/FreeSpaceDetect/blob/master/image/segment.jpg" width="45%">
</center>

Then convert the rest of point cloud to BEV view and after a set of image processing techniques we extract the edge points of the free zone and fit them with two smooth curves.

<center>
    <img src="https://github.com/tiger20/FreeSpaceDetect/blob/master/image/bev1.jpg" width="45%">
</center>

Also we can project the curve onto the image with the knonw calibration parameters.

<center>
    <img src="https://github.com/tiger20/FreeSpaceDetect/blob/master/image/freespace1.jpg" width="60%">
</center>
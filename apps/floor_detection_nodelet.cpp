// SPDX-License-Identifier: BSD-2-Clause

#include <memory>
#include <iostream>

#include <boost/optional.hpp>

#include <ros/ros.h>
#include <ros/time.h>
#include <pcl_ros/point_cloud.h>

#include <std_msgs/Time.h>
#include <sensor_msgs/PointCloud2.h>
#include <hdl_graph_slam/FloorCoeffs.h>//self-defined message

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/filters/impl/plane_clipper3D.hpp>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>

namespace hdl_graph_slam {

class FloorDetectionNodelet : public nodelet::Nodelet {
public:
  typedef pcl::PointXYZI PointT;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW //重载new函数，内存对齐

  FloorDetectionNodelet() {}
  virtual ~FloorDetectionNodelet() {}

  virtual void onInit() {
    NODELET_DEBUG("initializing floor_detection_nodelet...");
    nh = getNodeHandle();
    private_nh = getPrivateNodeHandle();

    initialize_params();

    points_sub = nh.subscribe("/filtered_points", 256, &FloorDetectionNodelet::cloud_callback, this);
    floor_pub = nh.advertise<hdl_graph_slam::FloorCoeffs>("/floor_detection/floor_coeffs", 32);

    read_until_pub = nh.advertise<std_msgs::Header>("/floor_detection/read_until", 32);
    floor_filtered_pub = nh.advertise<sensor_msgs::PointCloud2>("/floor_detection/floor_filtered_points", 32);
    floor_points_pub = nh.advertise<sensor_msgs::PointCloud2>("/floor_detection/floor_points", 32);
  }

private:
  /**
   * @brief initialize parameters
   */
  void initialize_params() {
    tilt_deg = private_nh.param<double>("tilt_deg", 0.0);                           // approximate sensor tilt angle [deg]
    sensor_height = private_nh.param<double>("sensor_height", 2.0);                 // approximate sensor height [m]
    height_clip_range = private_nh.param<double>("height_clip_range", 1.0);         // points with heights in [sensor_height - height_clip_range, sensor_height + height_clip_range] will be used for floor detection
    floor_pts_thresh = private_nh.param<int>("floor_pts_thresh", 512);              // minimum number of support points of RANSAC to accept a detected floor plane
    floor_normal_thresh = private_nh.param<double>("floor_normal_thresh", 10.0);    // verticality check thresold for the detected floor plane [deg]
    use_normal_filtering = private_nh.param<bool>("use_normal_filtering", true);    // if true, points with "non-"vertical normals will be filtered before RANSAC
    normal_filter_thresh = private_nh.param<double>("normal_filter_thresh", 20.0);  // "non-"verticality check threshold [deg]

    points_topic = private_nh.param<std::string>("points_topic", "/velodyne_points");
  }

  /**
   * @brief callback for point clouds
   * @param cloud_msg  point cloud msg
   */
  void cloud_callback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*cloud_msg, *cloud);

    if(cloud->empty()) {
      return;
    }

    // floor detection
    boost::optional<Eigen::Vector4f> floor = detect(cloud); //boost::optional 用于返回值的选择

    // publish the detected floor coefficients
    hdl_graph_slam::FloorCoeffs coeffs;
    coeffs.header = cloud_msg->header;
    if(floor) {
      coeffs.coeffs.resize(4);
      for(int i = 0; i < 4; i++) {
        coeffs.coeffs[i] = (*floor)[i];
      }
    }

    floor_pub.publish(coeffs);

    // for offline estimation
    std_msgs::HeaderPtr read_until(new std_msgs::Header());
    read_until->frame_id = points_topic;
    read_until->stamp = cloud_msg->header.stamp + ros::Duration(1, 0);//ros::Duration::Duration(uint32_t _sec, uint32_t _nsec)
    read_until_pub.publish(read_until);

    read_until->frame_id = "/filtered_points";
    read_until_pub.publish(read_until);
  }

  /**
   * @brief detect the floor plane from a point cloud
   * @param cloud  input cloud
   * @return detected floor plane coefficients
   */
  boost::optional<Eigen::Vector4f> detect(const pcl::PointCloud<PointT>::Ptr& cloud) const {
    // compensate the tilt rotation
    Eigen::Matrix4f tilt_matrix = Eigen::Matrix4f::Identity();//单位阵I
    tilt_matrix.topLeftCorner(3, 3) = Eigen::AngleAxisf(tilt_deg * M_PI / 180.0f, Eigen::Vector3f::UnitY()).toRotationMatrix();//Y轴转角度.在安装中为roll

    // filtering before RANSAC (height and normal filtering)
    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>);
    pcl::transformPointCloud(*cloud, *filtered, tilt_matrix);//tilt_deg=0无所谓，但这里的变换矩阵应该用逆。
    filtered = plane_clip(filtered, Eigen::Vector4f(0.0f, 0.0f, 1.0f, sensor_height + height_clip_range), false);//输入点云，平面参数，分割方向（false为下，true为上）
    filtered = plane_clip(filtered, Eigen::Vector4f(0.0f, 0.0f, 1.0f, sensor_height - height_clip_range), true);//vector4f(nx,ny,nz,d):单位法向量+距离原点距离

    if(use_normal_filtering) {
      filtered = normal_filtering(filtered);
    }

    pcl::transformPointCloud(*filtered, *filtered, static_cast<Eigen::Matrix4f>(tilt_matrix.inverse()));//去倾斜

    if(floor_filtered_pub.getNumSubscribers()) {
      filtered->header = cloud->header;
      floor_filtered_pub.publish(*filtered);
    }

    // too few points for RANSAC
    if(filtered->size() < floor_pts_thresh) {
      return boost::none;  //是一个类似空指针的none_t类型常量，表示未初始化
    }

    // RANSAC
    pcl::SampleConsensusModelPlane<PointT>::Ptr model_p(new pcl::SampleConsensusModelPlane<PointT>(filtered));
    pcl::RandomSampleConsensus<PointT> ransac(model_p);// RANSAC 拟合平面    
    ransac.setDistanceThreshold(0.1); // 设置内点阈值
    ransac.computeModel();

    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    ransac.getInliers(inliers->indices);// 获取内点   

    // too few inliers
    if(inliers->indices.size() < floor_pts_thresh) {
      return boost::none;
    }

    // verticality check of the detected floor's normal
    Eigen::Vector4f reference = tilt_matrix.inverse() * Eigen::Vector4f::UnitZ();

    EigenEigen::VectorXf coeffs;
    ransac.getModelCoefficients(coeffs); // 获取地面在激光雷达中的系数

    double dot = coeffs.head<3>().dot(reference.head<3>());
    if(std::abs(dot) < std::cos(floor_normal_thresh * M_PI / 180.0)) {
      // the normal is not vertical
      return boost::none;
    }

    // make the normal upward
    if(coeffs.head<3>().dot(Eigen::Vector3f::UnitZ()) < 0.0f) {
      coeffs *= -1.0f;
    }

    if(floor_points_pub.getNumSubscribers()) {
      pcl::PointCloud<PointT>::Ptr inlier_cloud(new pcl::PointCloud<PointT>);
      pcl::ExtractIndices<PointT> extract;
      extract.setInputCloud(filtered);
      extract.setIndices(inliers);
      extract.filter(*inlier_cloud);
      inlier_cloud->header = cloud->header;

      floor_points_pub.publish(*inlier_cloud);
    }

    return Eigen::Vector4f(coeffs);
  }

  /**
   * @brief plane_clip
   * @param src_cloud
   * @param plane
   * @param negative
   * @return
   */
  pcl::PointCloud<PointT>::Ptr plane_clip(const pcl::PointCloud<PointT>::Ptr& src_cloud, const Eigen::Vector4f& plane, bool negative) const {
    pcl::PlaneClipper3D<PointT> clipper(plane);
    pcl::PointIndices::Ptr indices(new pcl::PointIndices);

    clipper.clipPointCloud3D(*src_cloud, indices->indices);

    pcl::PointCloud<PointT>::Ptr dst_cloud(new pcl::PointCloud<PointT>);

    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(src_cloud);
    extract.setIndices(indices);
    extract.setNegative(negative);
    extract.filter(*dst_cloud);

    return dst_cloud;
  }

  /**
   * @brief filter points with non-vertical normals
   * @param cloud  input cloud
   * @return filtered cloud
   */
  pcl::PointCloud<PointT>::Ptr normal_filtering(const pcl::PointCloud<PointT>::Ptr& cloud) const {
    pcl::NormalEstimation<PointT, pcl::Normal> ne;
    ne.setInputCloud(cloud);

    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    ne.setSearchMethod(tree);//构建kdtree

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    ne.setKSearch(10);//点云法向计算时，需要搜索的近邻点大小
    ne.setViewPoint(0.0f, 0.0f, sensor_height);//视点的位置
    ne.compute(*normals);

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>);
    filtered->reserve(cloud->size());

    for(int i = 0; i < cloud->size(); i++) {
      float dot = normals->at(i).getNormalVector3fMap().normalized().dot(Eigen::Vector3f::UnitZ());//单位法向量与Z轴单位向量点积（向量间内积，a·b=|a||b|cos<a,b>）
      if(std::abs(dot) > std::cos(normal_filter_thresh * M_PI / 180.0)) { //负相关
        filtered->push_back(cloud->at(i));
      }
    }

    filtered->width = filtered->size();
    filtered->height = 1;
    filtered->is_dense = false;

    return filtered;
  }

private:
  ros::NodeHandle nh;
  ros::NodeHandle private_nh;

  // ROS topics
  ros::Subscriber points_sub;

  ros::Publisher floor_pub;
  ros::Publisher floor_points_pub;
  ros::Publisher floor_filtered_pub;

  std::string points_topic;
  ros::Publisher read_until_pub;

  // floor detection parameters
  // see initialize_params() for the details
  double tilt_deg;
  double sensor_height;
  double height_clip_range;

  int floor_pts_thresh;
  double floor_normal_thresh;

  bool use_normal_filtering;
  double normal_filter_thresh;
};

}  // namespace hdl_graph_slam

PLUGINLIB_EXPORT_CLASS(hdl_graph_slam::FloorDetectionNodelet, nodelet::Nodelet)

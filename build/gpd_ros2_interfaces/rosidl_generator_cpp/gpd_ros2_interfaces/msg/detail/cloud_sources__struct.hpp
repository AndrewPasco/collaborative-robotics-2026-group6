// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from gpd_ros2_interfaces:msg/CloudSources.idl
// generated code does not contain a copyright notice

#ifndef GPD_ROS2_INTERFACES__MSG__DETAIL__CLOUD_SOURCES__STRUCT_HPP_
#define GPD_ROS2_INTERFACES__MSG__DETAIL__CLOUD_SOURCES__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'cloud'
#include "sensor_msgs/msg/detail/point_cloud2__struct.hpp"
// Member 'view_points'
#include "geometry_msgs/msg/detail/point__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__gpd_ros2_interfaces__msg__CloudSources __attribute__((deprecated))
#else
# define DEPRECATED__gpd_ros2_interfaces__msg__CloudSources __declspec(deprecated)
#endif

namespace gpd_ros2_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct CloudSources_
{
  using Type = CloudSources_<ContainerAllocator>;

  explicit CloudSources_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : cloud(_init)
  {
    (void)_init;
  }

  explicit CloudSources_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : cloud(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _cloud_type =
    sensor_msgs::msg::PointCloud2_<ContainerAllocator>;
  _cloud_type cloud;
  using _camera_source_type =
    std::vector<int64_t, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<int64_t>>;
  _camera_source_type camera_source;
  using _view_points_type =
    std::vector<geometry_msgs::msg::Point_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<geometry_msgs::msg::Point_<ContainerAllocator>>>;
  _view_points_type view_points;

  // setters for named parameter idiom
  Type & set__cloud(
    const sensor_msgs::msg::PointCloud2_<ContainerAllocator> & _arg)
  {
    this->cloud = _arg;
    return *this;
  }
  Type & set__camera_source(
    const std::vector<int64_t, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<int64_t>> & _arg)
  {
    this->camera_source = _arg;
    return *this;
  }
  Type & set__view_points(
    const std::vector<geometry_msgs::msg::Point_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<geometry_msgs::msg::Point_<ContainerAllocator>>> & _arg)
  {
    this->view_points = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    gpd_ros2_interfaces::msg::CloudSources_<ContainerAllocator> *;
  using ConstRawPtr =
    const gpd_ros2_interfaces::msg::CloudSources_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<gpd_ros2_interfaces::msg::CloudSources_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<gpd_ros2_interfaces::msg::CloudSources_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      gpd_ros2_interfaces::msg::CloudSources_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<gpd_ros2_interfaces::msg::CloudSources_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      gpd_ros2_interfaces::msg::CloudSources_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<gpd_ros2_interfaces::msg::CloudSources_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<gpd_ros2_interfaces::msg::CloudSources_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<gpd_ros2_interfaces::msg::CloudSources_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__gpd_ros2_interfaces__msg__CloudSources
    std::shared_ptr<gpd_ros2_interfaces::msg::CloudSources_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__gpd_ros2_interfaces__msg__CloudSources
    std::shared_ptr<gpd_ros2_interfaces::msg::CloudSources_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const CloudSources_ & other) const
  {
    if (this->cloud != other.cloud) {
      return false;
    }
    if (this->camera_source != other.camera_source) {
      return false;
    }
    if (this->view_points != other.view_points) {
      return false;
    }
    return true;
  }
  bool operator!=(const CloudSources_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct CloudSources_

// alias to use template instance with default allocator
using CloudSources =
  gpd_ros2_interfaces::msg::CloudSources_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace gpd_ros2_interfaces

#endif  // GPD_ROS2_INTERFACES__MSG__DETAIL__CLOUD_SOURCES__STRUCT_HPP_

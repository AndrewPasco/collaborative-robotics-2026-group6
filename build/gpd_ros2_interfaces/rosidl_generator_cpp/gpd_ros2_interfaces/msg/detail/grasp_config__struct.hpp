// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from gpd_ros2_interfaces:msg/GraspConfig.idl
// generated code does not contain a copyright notice

#ifndef GPD_ROS2_INTERFACES__MSG__DETAIL__GRASP_CONFIG__STRUCT_HPP_
#define GPD_ROS2_INTERFACES__MSG__DETAIL__GRASP_CONFIG__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'position'
// Member 'sample'
#include "geometry_msgs/msg/detail/point__struct.hpp"
// Member 'approach'
// Member 'binormal'
// Member 'axis'
#include "geometry_msgs/msg/detail/vector3__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__gpd_ros2_interfaces__msg__GraspConfig __attribute__((deprecated))
#else
# define DEPRECATED__gpd_ros2_interfaces__msg__GraspConfig __declspec(deprecated)
#endif

namespace gpd_ros2_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct GraspConfig_
{
  using Type = GraspConfig_<ContainerAllocator>;

  explicit GraspConfig_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : position(_init),
    approach(_init),
    binormal(_init),
    axis(_init),
    sample(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->width = 0.0f;
      this->score = 0.0f;
    }
  }

  explicit GraspConfig_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : position(_alloc, _init),
    approach(_alloc, _init),
    binormal(_alloc, _init),
    axis(_alloc, _init),
    sample(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->width = 0.0f;
      this->score = 0.0f;
    }
  }

  // field types and members
  using _position_type =
    geometry_msgs::msg::Point_<ContainerAllocator>;
  _position_type position;
  using _approach_type =
    geometry_msgs::msg::Vector3_<ContainerAllocator>;
  _approach_type approach;
  using _binormal_type =
    geometry_msgs::msg::Vector3_<ContainerAllocator>;
  _binormal_type binormal;
  using _axis_type =
    geometry_msgs::msg::Vector3_<ContainerAllocator>;
  _axis_type axis;
  using _width_type =
    float;
  _width_type width;
  using _score_type =
    float;
  _score_type score;
  using _sample_type =
    geometry_msgs::msg::Point_<ContainerAllocator>;
  _sample_type sample;

  // setters for named parameter idiom
  Type & set__position(
    const geometry_msgs::msg::Point_<ContainerAllocator> & _arg)
  {
    this->position = _arg;
    return *this;
  }
  Type & set__approach(
    const geometry_msgs::msg::Vector3_<ContainerAllocator> & _arg)
  {
    this->approach = _arg;
    return *this;
  }
  Type & set__binormal(
    const geometry_msgs::msg::Vector3_<ContainerAllocator> & _arg)
  {
    this->binormal = _arg;
    return *this;
  }
  Type & set__axis(
    const geometry_msgs::msg::Vector3_<ContainerAllocator> & _arg)
  {
    this->axis = _arg;
    return *this;
  }
  Type & set__width(
    const float & _arg)
  {
    this->width = _arg;
    return *this;
  }
  Type & set__score(
    const float & _arg)
  {
    this->score = _arg;
    return *this;
  }
  Type & set__sample(
    const geometry_msgs::msg::Point_<ContainerAllocator> & _arg)
  {
    this->sample = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    gpd_ros2_interfaces::msg::GraspConfig_<ContainerAllocator> *;
  using ConstRawPtr =
    const gpd_ros2_interfaces::msg::GraspConfig_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<gpd_ros2_interfaces::msg::GraspConfig_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<gpd_ros2_interfaces::msg::GraspConfig_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      gpd_ros2_interfaces::msg::GraspConfig_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<gpd_ros2_interfaces::msg::GraspConfig_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      gpd_ros2_interfaces::msg::GraspConfig_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<gpd_ros2_interfaces::msg::GraspConfig_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<gpd_ros2_interfaces::msg::GraspConfig_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<gpd_ros2_interfaces::msg::GraspConfig_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__gpd_ros2_interfaces__msg__GraspConfig
    std::shared_ptr<gpd_ros2_interfaces::msg::GraspConfig_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__gpd_ros2_interfaces__msg__GraspConfig
    std::shared_ptr<gpd_ros2_interfaces::msg::GraspConfig_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const GraspConfig_ & other) const
  {
    if (this->position != other.position) {
      return false;
    }
    if (this->approach != other.approach) {
      return false;
    }
    if (this->binormal != other.binormal) {
      return false;
    }
    if (this->axis != other.axis) {
      return false;
    }
    if (this->width != other.width) {
      return false;
    }
    if (this->score != other.score) {
      return false;
    }
    if (this->sample != other.sample) {
      return false;
    }
    return true;
  }
  bool operator!=(const GraspConfig_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct GraspConfig_

// alias to use template instance with default allocator
using GraspConfig =
  gpd_ros2_interfaces::msg::GraspConfig_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace gpd_ros2_interfaces

#endif  // GPD_ROS2_INTERFACES__MSG__DETAIL__GRASP_CONFIG__STRUCT_HPP_

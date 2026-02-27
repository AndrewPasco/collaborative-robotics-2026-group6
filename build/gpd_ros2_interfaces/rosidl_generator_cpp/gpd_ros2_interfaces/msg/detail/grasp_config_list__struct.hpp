// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from gpd_ros2_interfaces:msg/GraspConfigList.idl
// generated code does not contain a copyright notice

#ifndef GPD_ROS2_INTERFACES__MSG__DETAIL__GRASP_CONFIG_LIST__STRUCT_HPP_
#define GPD_ROS2_INTERFACES__MSG__DETAIL__GRASP_CONFIG_LIST__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.hpp"
// Member 'grasps'
#include "gpd_ros2_interfaces/msg/detail/grasp_config__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__gpd_ros2_interfaces__msg__GraspConfigList __attribute__((deprecated))
#else
# define DEPRECATED__gpd_ros2_interfaces__msg__GraspConfigList __declspec(deprecated)
#endif

namespace gpd_ros2_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct GraspConfigList_
{
  using Type = GraspConfigList_<ContainerAllocator>;

  explicit GraspConfigList_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    (void)_init;
  }

  explicit GraspConfigList_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _grasps_type =
    std::vector<gpd_ros2_interfaces::msg::GraspConfig_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<gpd_ros2_interfaces::msg::GraspConfig_<ContainerAllocator>>>;
  _grasps_type grasps;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__grasps(
    const std::vector<gpd_ros2_interfaces::msg::GraspConfig_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<gpd_ros2_interfaces::msg::GraspConfig_<ContainerAllocator>>> & _arg)
  {
    this->grasps = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    gpd_ros2_interfaces::msg::GraspConfigList_<ContainerAllocator> *;
  using ConstRawPtr =
    const gpd_ros2_interfaces::msg::GraspConfigList_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<gpd_ros2_interfaces::msg::GraspConfigList_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<gpd_ros2_interfaces::msg::GraspConfigList_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      gpd_ros2_interfaces::msg::GraspConfigList_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<gpd_ros2_interfaces::msg::GraspConfigList_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      gpd_ros2_interfaces::msg::GraspConfigList_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<gpd_ros2_interfaces::msg::GraspConfigList_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<gpd_ros2_interfaces::msg::GraspConfigList_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<gpd_ros2_interfaces::msg::GraspConfigList_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__gpd_ros2_interfaces__msg__GraspConfigList
    std::shared_ptr<gpd_ros2_interfaces::msg::GraspConfigList_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__gpd_ros2_interfaces__msg__GraspConfigList
    std::shared_ptr<gpd_ros2_interfaces::msg::GraspConfigList_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const GraspConfigList_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->grasps != other.grasps) {
      return false;
    }
    return true;
  }
  bool operator!=(const GraspConfigList_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct GraspConfigList_

// alias to use template instance with default allocator
using GraspConfigList =
  gpd_ros2_interfaces::msg::GraspConfigList_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace gpd_ros2_interfaces

#endif  // GPD_ROS2_INTERFACES__MSG__DETAIL__GRASP_CONFIG_LIST__STRUCT_HPP_

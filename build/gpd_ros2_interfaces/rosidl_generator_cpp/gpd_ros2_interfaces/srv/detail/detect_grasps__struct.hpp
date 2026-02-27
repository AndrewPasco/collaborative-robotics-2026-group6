// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from gpd_ros2_interfaces:srv/DetectGrasps.idl
// generated code does not contain a copyright notice

#ifndef GPD_ROS2_INTERFACES__SRV__DETAIL__DETECT_GRASPS__STRUCT_HPP_
#define GPD_ROS2_INTERFACES__SRV__DETAIL__DETECT_GRASPS__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'cloud_indexed'
#include "gpd_ros2_interfaces/msg/detail/cloud_indexed__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__gpd_ros2_interfaces__srv__DetectGrasps_Request __attribute__((deprecated))
#else
# define DEPRECATED__gpd_ros2_interfaces__srv__DetectGrasps_Request __declspec(deprecated)
#endif

namespace gpd_ros2_interfaces
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct DetectGrasps_Request_
{
  using Type = DetectGrasps_Request_<ContainerAllocator>;

  explicit DetectGrasps_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : cloud_indexed(_init)
  {
    (void)_init;
  }

  explicit DetectGrasps_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : cloud_indexed(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _cloud_indexed_type =
    gpd_ros2_interfaces::msg::CloudIndexed_<ContainerAllocator>;
  _cloud_indexed_type cloud_indexed;

  // setters for named parameter idiom
  Type & set__cloud_indexed(
    const gpd_ros2_interfaces::msg::CloudIndexed_<ContainerAllocator> & _arg)
  {
    this->cloud_indexed = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    gpd_ros2_interfaces::srv::DetectGrasps_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const gpd_ros2_interfaces::srv::DetectGrasps_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<gpd_ros2_interfaces::srv::DetectGrasps_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<gpd_ros2_interfaces::srv::DetectGrasps_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      gpd_ros2_interfaces::srv::DetectGrasps_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<gpd_ros2_interfaces::srv::DetectGrasps_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      gpd_ros2_interfaces::srv::DetectGrasps_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<gpd_ros2_interfaces::srv::DetectGrasps_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<gpd_ros2_interfaces::srv::DetectGrasps_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<gpd_ros2_interfaces::srv::DetectGrasps_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__gpd_ros2_interfaces__srv__DetectGrasps_Request
    std::shared_ptr<gpd_ros2_interfaces::srv::DetectGrasps_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__gpd_ros2_interfaces__srv__DetectGrasps_Request
    std::shared_ptr<gpd_ros2_interfaces::srv::DetectGrasps_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const DetectGrasps_Request_ & other) const
  {
    if (this->cloud_indexed != other.cloud_indexed) {
      return false;
    }
    return true;
  }
  bool operator!=(const DetectGrasps_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct DetectGrasps_Request_

// alias to use template instance with default allocator
using DetectGrasps_Request =
  gpd_ros2_interfaces::srv::DetectGrasps_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace gpd_ros2_interfaces


// Include directives for member types
// Member 'grasp_configs'
#include "gpd_ros2_interfaces/msg/detail/grasp_config_list__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__gpd_ros2_interfaces__srv__DetectGrasps_Response __attribute__((deprecated))
#else
# define DEPRECATED__gpd_ros2_interfaces__srv__DetectGrasps_Response __declspec(deprecated)
#endif

namespace gpd_ros2_interfaces
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct DetectGrasps_Response_
{
  using Type = DetectGrasps_Response_<ContainerAllocator>;

  explicit DetectGrasps_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : grasp_configs(_init)
  {
    (void)_init;
  }

  explicit DetectGrasps_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : grasp_configs(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _grasp_configs_type =
    gpd_ros2_interfaces::msg::GraspConfigList_<ContainerAllocator>;
  _grasp_configs_type grasp_configs;

  // setters for named parameter idiom
  Type & set__grasp_configs(
    const gpd_ros2_interfaces::msg::GraspConfigList_<ContainerAllocator> & _arg)
  {
    this->grasp_configs = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    gpd_ros2_interfaces::srv::DetectGrasps_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const gpd_ros2_interfaces::srv::DetectGrasps_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<gpd_ros2_interfaces::srv::DetectGrasps_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<gpd_ros2_interfaces::srv::DetectGrasps_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      gpd_ros2_interfaces::srv::DetectGrasps_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<gpd_ros2_interfaces::srv::DetectGrasps_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      gpd_ros2_interfaces::srv::DetectGrasps_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<gpd_ros2_interfaces::srv::DetectGrasps_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<gpd_ros2_interfaces::srv::DetectGrasps_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<gpd_ros2_interfaces::srv::DetectGrasps_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__gpd_ros2_interfaces__srv__DetectGrasps_Response
    std::shared_ptr<gpd_ros2_interfaces::srv::DetectGrasps_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__gpd_ros2_interfaces__srv__DetectGrasps_Response
    std::shared_ptr<gpd_ros2_interfaces::srv::DetectGrasps_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const DetectGrasps_Response_ & other) const
  {
    if (this->grasp_configs != other.grasp_configs) {
      return false;
    }
    return true;
  }
  bool operator!=(const DetectGrasps_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct DetectGrasps_Response_

// alias to use template instance with default allocator
using DetectGrasps_Response =
  gpd_ros2_interfaces::srv::DetectGrasps_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace gpd_ros2_interfaces

namespace gpd_ros2_interfaces
{

namespace srv
{

struct DetectGrasps
{
  using Request = gpd_ros2_interfaces::srv::DetectGrasps_Request;
  using Response = gpd_ros2_interfaces::srv::DetectGrasps_Response;
};

}  // namespace srv

}  // namespace gpd_ros2_interfaces

#endif  // GPD_ROS2_INTERFACES__SRV__DETAIL__DETECT_GRASPS__STRUCT_HPP_

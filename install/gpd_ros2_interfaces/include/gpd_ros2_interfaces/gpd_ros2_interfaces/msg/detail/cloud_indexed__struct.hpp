// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from gpd_ros2_interfaces:msg/CloudIndexed.idl
// generated code does not contain a copyright notice

#ifndef GPD_ROS2_INTERFACES__MSG__DETAIL__CLOUD_INDEXED__STRUCT_HPP_
#define GPD_ROS2_INTERFACES__MSG__DETAIL__CLOUD_INDEXED__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'cloud_sources'
#include "gpd_ros2_interfaces/msg/detail/cloud_sources__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__gpd_ros2_interfaces__msg__CloudIndexed __attribute__((deprecated))
#else
# define DEPRECATED__gpd_ros2_interfaces__msg__CloudIndexed __declspec(deprecated)
#endif

namespace gpd_ros2_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct CloudIndexed_
{
  using Type = CloudIndexed_<ContainerAllocator>;

  explicit CloudIndexed_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : cloud_sources(_init)
  {
    (void)_init;
  }

  explicit CloudIndexed_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : cloud_sources(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _cloud_sources_type =
    gpd_ros2_interfaces::msg::CloudSources_<ContainerAllocator>;
  _cloud_sources_type cloud_sources;
  using _indices_type =
    std::vector<int64_t, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<int64_t>>;
  _indices_type indices;

  // setters for named parameter idiom
  Type & set__cloud_sources(
    const gpd_ros2_interfaces::msg::CloudSources_<ContainerAllocator> & _arg)
  {
    this->cloud_sources = _arg;
    return *this;
  }
  Type & set__indices(
    const std::vector<int64_t, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<int64_t>> & _arg)
  {
    this->indices = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    gpd_ros2_interfaces::msg::CloudIndexed_<ContainerAllocator> *;
  using ConstRawPtr =
    const gpd_ros2_interfaces::msg::CloudIndexed_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<gpd_ros2_interfaces::msg::CloudIndexed_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<gpd_ros2_interfaces::msg::CloudIndexed_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      gpd_ros2_interfaces::msg::CloudIndexed_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<gpd_ros2_interfaces::msg::CloudIndexed_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      gpd_ros2_interfaces::msg::CloudIndexed_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<gpd_ros2_interfaces::msg::CloudIndexed_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<gpd_ros2_interfaces::msg::CloudIndexed_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<gpd_ros2_interfaces::msg::CloudIndexed_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__gpd_ros2_interfaces__msg__CloudIndexed
    std::shared_ptr<gpd_ros2_interfaces::msg::CloudIndexed_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__gpd_ros2_interfaces__msg__CloudIndexed
    std::shared_ptr<gpd_ros2_interfaces::msg::CloudIndexed_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const CloudIndexed_ & other) const
  {
    if (this->cloud_sources != other.cloud_sources) {
      return false;
    }
    if (this->indices != other.indices) {
      return false;
    }
    return true;
  }
  bool operator!=(const CloudIndexed_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct CloudIndexed_

// alias to use template instance with default allocator
using CloudIndexed =
  gpd_ros2_interfaces::msg::CloudIndexed_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace gpd_ros2_interfaces

#endif  // GPD_ROS2_INTERFACES__MSG__DETAIL__CLOUD_INDEXED__STRUCT_HPP_

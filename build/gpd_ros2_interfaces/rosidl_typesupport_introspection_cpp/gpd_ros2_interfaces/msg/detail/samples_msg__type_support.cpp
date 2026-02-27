// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from gpd_ros2_interfaces:msg/SamplesMsg.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "gpd_ros2_interfaces/msg/detail/samples_msg__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace gpd_ros2_interfaces
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void SamplesMsg_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) gpd_ros2_interfaces::msg::SamplesMsg(_init);
}

void SamplesMsg_fini_function(void * message_memory)
{
  auto typed_message = static_cast<gpd_ros2_interfaces::msg::SamplesMsg *>(message_memory);
  typed_message->~SamplesMsg();
}

size_t size_function__SamplesMsg__samples(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<geometry_msgs::msg::Point> *>(untyped_member);
  return member->size();
}

const void * get_const_function__SamplesMsg__samples(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<geometry_msgs::msg::Point> *>(untyped_member);
  return &member[index];
}

void * get_function__SamplesMsg__samples(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<geometry_msgs::msg::Point> *>(untyped_member);
  return &member[index];
}

void fetch_function__SamplesMsg__samples(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const geometry_msgs::msg::Point *>(
    get_const_function__SamplesMsg__samples(untyped_member, index));
  auto & value = *reinterpret_cast<geometry_msgs::msg::Point *>(untyped_value);
  value = item;
}

void assign_function__SamplesMsg__samples(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<geometry_msgs::msg::Point *>(
    get_function__SamplesMsg__samples(untyped_member, index));
  const auto & value = *reinterpret_cast<const geometry_msgs::msg::Point *>(untyped_value);
  item = value;
}

void resize_function__SamplesMsg__samples(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<geometry_msgs::msg::Point> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember SamplesMsg_message_member_array[2] = {
  {
    "header",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<std_msgs::msg::Header>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(gpd_ros2_interfaces::msg::SamplesMsg, header),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "samples",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<geometry_msgs::msg::Point>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(gpd_ros2_interfaces::msg::SamplesMsg, samples),  // bytes offset in struct
    nullptr,  // default value
    size_function__SamplesMsg__samples,  // size() function pointer
    get_const_function__SamplesMsg__samples,  // get_const(index) function pointer
    get_function__SamplesMsg__samples,  // get(index) function pointer
    fetch_function__SamplesMsg__samples,  // fetch(index, &value) function pointer
    assign_function__SamplesMsg__samples,  // assign(index, value) function pointer
    resize_function__SamplesMsg__samples  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers SamplesMsg_message_members = {
  "gpd_ros2_interfaces::msg",  // message namespace
  "SamplesMsg",  // message name
  2,  // number of fields
  sizeof(gpd_ros2_interfaces::msg::SamplesMsg),
  SamplesMsg_message_member_array,  // message members
  SamplesMsg_init_function,  // function to initialize message memory (memory has to be allocated)
  SamplesMsg_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t SamplesMsg_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &SamplesMsg_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace gpd_ros2_interfaces


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<gpd_ros2_interfaces::msg::SamplesMsg>()
{
  return &::gpd_ros2_interfaces::msg::rosidl_typesupport_introspection_cpp::SamplesMsg_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, gpd_ros2_interfaces, msg, SamplesMsg)() {
  return &::gpd_ros2_interfaces::msg::rosidl_typesupport_introspection_cpp::SamplesMsg_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

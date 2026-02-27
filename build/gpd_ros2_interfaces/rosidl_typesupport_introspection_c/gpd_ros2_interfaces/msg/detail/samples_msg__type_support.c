// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from gpd_ros2_interfaces:msg/SamplesMsg.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "gpd_ros2_interfaces/msg/detail/samples_msg__rosidl_typesupport_introspection_c.h"
#include "gpd_ros2_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "gpd_ros2_interfaces/msg/detail/samples_msg__functions.h"
#include "gpd_ros2_interfaces/msg/detail/samples_msg__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"
// Member `samples`
#include "geometry_msgs/msg/point.h"
// Member `samples`
#include "geometry_msgs/msg/detail/point__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void gpd_ros2_interfaces__msg__SamplesMsg__rosidl_typesupport_introspection_c__SamplesMsg_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  gpd_ros2_interfaces__msg__SamplesMsg__init(message_memory);
}

void gpd_ros2_interfaces__msg__SamplesMsg__rosidl_typesupport_introspection_c__SamplesMsg_fini_function(void * message_memory)
{
  gpd_ros2_interfaces__msg__SamplesMsg__fini(message_memory);
}

size_t gpd_ros2_interfaces__msg__SamplesMsg__rosidl_typesupport_introspection_c__size_function__SamplesMsg__samples(
  const void * untyped_member)
{
  const geometry_msgs__msg__Point__Sequence * member =
    (const geometry_msgs__msg__Point__Sequence *)(untyped_member);
  return member->size;
}

const void * gpd_ros2_interfaces__msg__SamplesMsg__rosidl_typesupport_introspection_c__get_const_function__SamplesMsg__samples(
  const void * untyped_member, size_t index)
{
  const geometry_msgs__msg__Point__Sequence * member =
    (const geometry_msgs__msg__Point__Sequence *)(untyped_member);
  return &member->data[index];
}

void * gpd_ros2_interfaces__msg__SamplesMsg__rosidl_typesupport_introspection_c__get_function__SamplesMsg__samples(
  void * untyped_member, size_t index)
{
  geometry_msgs__msg__Point__Sequence * member =
    (geometry_msgs__msg__Point__Sequence *)(untyped_member);
  return &member->data[index];
}

void gpd_ros2_interfaces__msg__SamplesMsg__rosidl_typesupport_introspection_c__fetch_function__SamplesMsg__samples(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const geometry_msgs__msg__Point * item =
    ((const geometry_msgs__msg__Point *)
    gpd_ros2_interfaces__msg__SamplesMsg__rosidl_typesupport_introspection_c__get_const_function__SamplesMsg__samples(untyped_member, index));
  geometry_msgs__msg__Point * value =
    (geometry_msgs__msg__Point *)(untyped_value);
  *value = *item;
}

void gpd_ros2_interfaces__msg__SamplesMsg__rosidl_typesupport_introspection_c__assign_function__SamplesMsg__samples(
  void * untyped_member, size_t index, const void * untyped_value)
{
  geometry_msgs__msg__Point * item =
    ((geometry_msgs__msg__Point *)
    gpd_ros2_interfaces__msg__SamplesMsg__rosidl_typesupport_introspection_c__get_function__SamplesMsg__samples(untyped_member, index));
  const geometry_msgs__msg__Point * value =
    (const geometry_msgs__msg__Point *)(untyped_value);
  *item = *value;
}

bool gpd_ros2_interfaces__msg__SamplesMsg__rosidl_typesupport_introspection_c__resize_function__SamplesMsg__samples(
  void * untyped_member, size_t size)
{
  geometry_msgs__msg__Point__Sequence * member =
    (geometry_msgs__msg__Point__Sequence *)(untyped_member);
  geometry_msgs__msg__Point__Sequence__fini(member);
  return geometry_msgs__msg__Point__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember gpd_ros2_interfaces__msg__SamplesMsg__rosidl_typesupport_introspection_c__SamplesMsg_message_member_array[2] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(gpd_ros2_interfaces__msg__SamplesMsg, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "samples",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(gpd_ros2_interfaces__msg__SamplesMsg, samples),  // bytes offset in struct
    NULL,  // default value
    gpd_ros2_interfaces__msg__SamplesMsg__rosidl_typesupport_introspection_c__size_function__SamplesMsg__samples,  // size() function pointer
    gpd_ros2_interfaces__msg__SamplesMsg__rosidl_typesupport_introspection_c__get_const_function__SamplesMsg__samples,  // get_const(index) function pointer
    gpd_ros2_interfaces__msg__SamplesMsg__rosidl_typesupport_introspection_c__get_function__SamplesMsg__samples,  // get(index) function pointer
    gpd_ros2_interfaces__msg__SamplesMsg__rosidl_typesupport_introspection_c__fetch_function__SamplesMsg__samples,  // fetch(index, &value) function pointer
    gpd_ros2_interfaces__msg__SamplesMsg__rosidl_typesupport_introspection_c__assign_function__SamplesMsg__samples,  // assign(index, value) function pointer
    gpd_ros2_interfaces__msg__SamplesMsg__rosidl_typesupport_introspection_c__resize_function__SamplesMsg__samples  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers gpd_ros2_interfaces__msg__SamplesMsg__rosidl_typesupport_introspection_c__SamplesMsg_message_members = {
  "gpd_ros2_interfaces__msg",  // message namespace
  "SamplesMsg",  // message name
  2,  // number of fields
  sizeof(gpd_ros2_interfaces__msg__SamplesMsg),
  gpd_ros2_interfaces__msg__SamplesMsg__rosidl_typesupport_introspection_c__SamplesMsg_message_member_array,  // message members
  gpd_ros2_interfaces__msg__SamplesMsg__rosidl_typesupport_introspection_c__SamplesMsg_init_function,  // function to initialize message memory (memory has to be allocated)
  gpd_ros2_interfaces__msg__SamplesMsg__rosidl_typesupport_introspection_c__SamplesMsg_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t gpd_ros2_interfaces__msg__SamplesMsg__rosidl_typesupport_introspection_c__SamplesMsg_message_type_support_handle = {
  0,
  &gpd_ros2_interfaces__msg__SamplesMsg__rosidl_typesupport_introspection_c__SamplesMsg_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_gpd_ros2_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, gpd_ros2_interfaces, msg, SamplesMsg)() {
  gpd_ros2_interfaces__msg__SamplesMsg__rosidl_typesupport_introspection_c__SamplesMsg_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  gpd_ros2_interfaces__msg__SamplesMsg__rosidl_typesupport_introspection_c__SamplesMsg_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Point)();
  if (!gpd_ros2_interfaces__msg__SamplesMsg__rosidl_typesupport_introspection_c__SamplesMsg_message_type_support_handle.typesupport_identifier) {
    gpd_ros2_interfaces__msg__SamplesMsg__rosidl_typesupport_introspection_c__SamplesMsg_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &gpd_ros2_interfaces__msg__SamplesMsg__rosidl_typesupport_introspection_c__SamplesMsg_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

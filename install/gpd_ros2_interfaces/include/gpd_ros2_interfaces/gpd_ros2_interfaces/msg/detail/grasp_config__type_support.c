// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from gpd_ros2_interfaces:msg/GraspConfig.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "gpd_ros2_interfaces/msg/detail/grasp_config__rosidl_typesupport_introspection_c.h"
#include "gpd_ros2_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "gpd_ros2_interfaces/msg/detail/grasp_config__functions.h"
#include "gpd_ros2_interfaces/msg/detail/grasp_config__struct.h"


// Include directives for member types
// Member `position`
// Member `sample`
#include "geometry_msgs/msg/point.h"
// Member `position`
// Member `sample`
#include "geometry_msgs/msg/detail/point__rosidl_typesupport_introspection_c.h"
// Member `approach`
// Member `binormal`
// Member `axis`
#include "geometry_msgs/msg/vector3.h"
// Member `approach`
// Member `binormal`
// Member `axis`
#include "geometry_msgs/msg/detail/vector3__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void gpd_ros2_interfaces__msg__GraspConfig__rosidl_typesupport_introspection_c__GraspConfig_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  gpd_ros2_interfaces__msg__GraspConfig__init(message_memory);
}

void gpd_ros2_interfaces__msg__GraspConfig__rosidl_typesupport_introspection_c__GraspConfig_fini_function(void * message_memory)
{
  gpd_ros2_interfaces__msg__GraspConfig__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember gpd_ros2_interfaces__msg__GraspConfig__rosidl_typesupport_introspection_c__GraspConfig_message_member_array[7] = {
  {
    "position",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(gpd_ros2_interfaces__msg__GraspConfig, position),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "approach",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(gpd_ros2_interfaces__msg__GraspConfig, approach),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "binormal",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(gpd_ros2_interfaces__msg__GraspConfig, binormal),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "axis",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(gpd_ros2_interfaces__msg__GraspConfig, axis),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "width",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(gpd_ros2_interfaces__msg__GraspConfig, width),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "score",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(gpd_ros2_interfaces__msg__GraspConfig, score),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "sample",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(gpd_ros2_interfaces__msg__GraspConfig, sample),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers gpd_ros2_interfaces__msg__GraspConfig__rosidl_typesupport_introspection_c__GraspConfig_message_members = {
  "gpd_ros2_interfaces__msg",  // message namespace
  "GraspConfig",  // message name
  7,  // number of fields
  sizeof(gpd_ros2_interfaces__msg__GraspConfig),
  gpd_ros2_interfaces__msg__GraspConfig__rosidl_typesupport_introspection_c__GraspConfig_message_member_array,  // message members
  gpd_ros2_interfaces__msg__GraspConfig__rosidl_typesupport_introspection_c__GraspConfig_init_function,  // function to initialize message memory (memory has to be allocated)
  gpd_ros2_interfaces__msg__GraspConfig__rosidl_typesupport_introspection_c__GraspConfig_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t gpd_ros2_interfaces__msg__GraspConfig__rosidl_typesupport_introspection_c__GraspConfig_message_type_support_handle = {
  0,
  &gpd_ros2_interfaces__msg__GraspConfig__rosidl_typesupport_introspection_c__GraspConfig_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_gpd_ros2_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, gpd_ros2_interfaces, msg, GraspConfig)() {
  gpd_ros2_interfaces__msg__GraspConfig__rosidl_typesupport_introspection_c__GraspConfig_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Point)();
  gpd_ros2_interfaces__msg__GraspConfig__rosidl_typesupport_introspection_c__GraspConfig_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Vector3)();
  gpd_ros2_interfaces__msg__GraspConfig__rosidl_typesupport_introspection_c__GraspConfig_message_member_array[2].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Vector3)();
  gpd_ros2_interfaces__msg__GraspConfig__rosidl_typesupport_introspection_c__GraspConfig_message_member_array[3].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Vector3)();
  gpd_ros2_interfaces__msg__GraspConfig__rosidl_typesupport_introspection_c__GraspConfig_message_member_array[6].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Point)();
  if (!gpd_ros2_interfaces__msg__GraspConfig__rosidl_typesupport_introspection_c__GraspConfig_message_type_support_handle.typesupport_identifier) {
    gpd_ros2_interfaces__msg__GraspConfig__rosidl_typesupport_introspection_c__GraspConfig_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &gpd_ros2_interfaces__msg__GraspConfig__rosidl_typesupport_introspection_c__GraspConfig_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

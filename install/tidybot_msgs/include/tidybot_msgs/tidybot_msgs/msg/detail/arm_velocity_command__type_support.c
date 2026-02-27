// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from tidybot_msgs:msg/ArmVelocityCommand.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "tidybot_msgs/msg/detail/arm_velocity_command__rosidl_typesupport_introspection_c.h"
#include "tidybot_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "tidybot_msgs/msg/detail/arm_velocity_command__functions.h"
#include "tidybot_msgs/msg/detail/arm_velocity_command__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void tidybot_msgs__msg__ArmVelocityCommand__rosidl_typesupport_introspection_c__ArmVelocityCommand_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  tidybot_msgs__msg__ArmVelocityCommand__init(message_memory);
}

void tidybot_msgs__msg__ArmVelocityCommand__rosidl_typesupport_introspection_c__ArmVelocityCommand_fini_function(void * message_memory)
{
  tidybot_msgs__msg__ArmVelocityCommand__fini(message_memory);
}

size_t tidybot_msgs__msg__ArmVelocityCommand__rosidl_typesupport_introspection_c__size_function__ArmVelocityCommand__joint_velocities(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * tidybot_msgs__msg__ArmVelocityCommand__rosidl_typesupport_introspection_c__get_const_function__ArmVelocityCommand__joint_velocities(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * tidybot_msgs__msg__ArmVelocityCommand__rosidl_typesupport_introspection_c__get_function__ArmVelocityCommand__joint_velocities(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void tidybot_msgs__msg__ArmVelocityCommand__rosidl_typesupport_introspection_c__fetch_function__ArmVelocityCommand__joint_velocities(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    tidybot_msgs__msg__ArmVelocityCommand__rosidl_typesupport_introspection_c__get_const_function__ArmVelocityCommand__joint_velocities(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void tidybot_msgs__msg__ArmVelocityCommand__rosidl_typesupport_introspection_c__assign_function__ArmVelocityCommand__joint_velocities(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    tidybot_msgs__msg__ArmVelocityCommand__rosidl_typesupport_introspection_c__get_function__ArmVelocityCommand__joint_velocities(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

static rosidl_typesupport_introspection_c__MessageMember tidybot_msgs__msg__ArmVelocityCommand__rosidl_typesupport_introspection_c__ArmVelocityCommand_message_member_array[3] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tidybot_msgs__msg__ArmVelocityCommand, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "joint_velocities",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(tidybot_msgs__msg__ArmVelocityCommand, joint_velocities),  // bytes offset in struct
    NULL,  // default value
    tidybot_msgs__msg__ArmVelocityCommand__rosidl_typesupport_introspection_c__size_function__ArmVelocityCommand__joint_velocities,  // size() function pointer
    tidybot_msgs__msg__ArmVelocityCommand__rosidl_typesupport_introspection_c__get_const_function__ArmVelocityCommand__joint_velocities,  // get_const(index) function pointer
    tidybot_msgs__msg__ArmVelocityCommand__rosidl_typesupport_introspection_c__get_function__ArmVelocityCommand__joint_velocities,  // get(index) function pointer
    tidybot_msgs__msg__ArmVelocityCommand__rosidl_typesupport_introspection_c__fetch_function__ArmVelocityCommand__joint_velocities,  // fetch(index, &value) function pointer
    tidybot_msgs__msg__ArmVelocityCommand__rosidl_typesupport_introspection_c__assign_function__ArmVelocityCommand__joint_velocities,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "duration",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tidybot_msgs__msg__ArmVelocityCommand, duration),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers tidybot_msgs__msg__ArmVelocityCommand__rosidl_typesupport_introspection_c__ArmVelocityCommand_message_members = {
  "tidybot_msgs__msg",  // message namespace
  "ArmVelocityCommand",  // message name
  3,  // number of fields
  sizeof(tidybot_msgs__msg__ArmVelocityCommand),
  tidybot_msgs__msg__ArmVelocityCommand__rosidl_typesupport_introspection_c__ArmVelocityCommand_message_member_array,  // message members
  tidybot_msgs__msg__ArmVelocityCommand__rosidl_typesupport_introspection_c__ArmVelocityCommand_init_function,  // function to initialize message memory (memory has to be allocated)
  tidybot_msgs__msg__ArmVelocityCommand__rosidl_typesupport_introspection_c__ArmVelocityCommand_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t tidybot_msgs__msg__ArmVelocityCommand__rosidl_typesupport_introspection_c__ArmVelocityCommand_message_type_support_handle = {
  0,
  &tidybot_msgs__msg__ArmVelocityCommand__rosidl_typesupport_introspection_c__ArmVelocityCommand_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_tidybot_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, tidybot_msgs, msg, ArmVelocityCommand)() {
  tidybot_msgs__msg__ArmVelocityCommand__rosidl_typesupport_introspection_c__ArmVelocityCommand_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  if (!tidybot_msgs__msg__ArmVelocityCommand__rosidl_typesupport_introspection_c__ArmVelocityCommand_message_type_support_handle.typesupport_identifier) {
    tidybot_msgs__msg__ArmVelocityCommand__rosidl_typesupport_introspection_c__ArmVelocityCommand_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &tidybot_msgs__msg__ArmVelocityCommand__rosidl_typesupport_introspection_c__ArmVelocityCommand_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
